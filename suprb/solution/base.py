from __future__ import annotations

import itertools
from abc import ABCMeta, abstractmethod
from typing import Union

import numpy as np
from sklearn.base import RegressorMixin, ClassifierMixin, BaseEstimator
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score

from suprb.rule import Rule
from suprb.base import BaseComponent, SolutionBase, SupervisedMixin
from suprb.fitness import BaseFitness, pseudo_error


class MixingModel(BaseComponent, metaclass=ABCMeta):
    """Performs mixing of local `Rule`s to obtain a complete prediction of the input space."""

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, X: np.ndarray, subpopulation: list[Rule], cache=False) -> np.ndarray:
        pass


class SolutionFitness(BaseFitness, metaclass=ABCMeta):
    """Evaluate the fitness of a `Solution`."""

    max_genome_length_: int

    @abstractmethod
    def __call__(self, solution: Solution) -> float:
        pass


class Solution(SolutionBase, SupervisedMixin):
    """Solution that mixes a subpopulation of rules."""

    input_size_: int
    complexity_: int

    def __init__(self, genome: np.ndarray, pool: list[Rule], mixing: MixingModel, fitness: SolutionFitness):
        self.genome = genome
        self.pool = pool
        self.mixing = mixing
        self.fitness = fitness
        self.isClass = None

    def score(self, X, y, sample_weight=None):
        if not self.pool:
            return 0.0
        if self.isClass is None:
            if isinstance(self.pool[0].model, ClassifierMixin):
                self.isClass = True
            else:
                self.isClass = False
        if self.isClass:
            return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
        else:
            y_pred = self.predict(X)
            return r2_score(y, y_pred, sample_weight=sample_weight)
            

    def fit(self, X: np.ndarray, y: np.ndarray) -> Solution:
        pred = self.predict(X, cache=True)
        if not self.pool:
            self.error_ = 9999
        else:
            if self.isClass is None:
                if isinstance(self.pool[0].model, ClassifierMixin):
                    self.isClass = True
                else:
                    self.isClass = False
            if self.isClass:
                self.error_ = max(pseudo_error(accuracy_score(y, pred)), 1e-4)
            else:
                self.error_ = max(mean_squared_error(y, pred), 1e-4)
            
        self.input_size_ = self.genome.shape[0]
        self.complexity_ = np.sum(self.genome).item()  # equivalent to np.count_nonzero, but possibly faster
        self.fitness_ = self.fitness(self)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray, cache=False) -> np.ndarray:
        """
        This function performs MoE of the rules in the subpopulation.

        The attribute `cache` decides if the prediction should use the cached data from the rules, which
        includes rule error, fitness, predictions and the binary match string.
        It is True while fitting, because the data is identical there and caching saves a good amount of time.
        For predictions after fitting, it is false, because all data needs to be recalculated from scratch.
        """

        return self.mixing(X=X, subpopulation=self.subpopulation, cache=cache)

    @property
    def subpopulation(self) -> list[Rule]:
        """Get all rules in the subpopulation."""
        assert (len(self.genome) == len(self.pool))
        return list(itertools.compress(self.pool, self.genome))

    def clone(self, **kwargs) -> Solution:
        args = dict(
            genome=self.genome.copy() if 'genome' not in kwargs else None,
            pool=self.pool,
            mixing=self.mixing,
            fitness=self.fitness
        )
        solution = Solution(**(args | kwargs))
        if not kwargs:
            attributes = ['fitness_', 'error_', 'complexity_', 'is_fitted_', 'input_size_']
            solution.__dict__ |= {key: getattr(self, key) for key in attributes}
        return solution

    def _more_str_attributes(self) -> dict:
        return {'complexity': self.complexity_}
