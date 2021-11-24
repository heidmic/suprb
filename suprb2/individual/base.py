from __future__ import annotations

import itertools
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error

from suprb2.rule import Rule
from suprb2.base import BaseComponent, Solution
from suprb2.fitness import BaseFitness


class MixingModel(BaseComponent, metaclass=ABCMeta):
    """Performs mixing of local `Rule`s to obtain a complete prediction of the input space."""

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, X: np.ndarray, subpopulation: list[Rule], cache=False) -> np.ndarray:
        pass


class IndividualFitness(BaseFitness, metaclass=ABCMeta):
    """Evaluate the fitness of a `Individual`."""

    @abstractmethod
    def __call__(self, individual: Individual) -> float:
        pass


class Individual(Solution):
    """Individual that mixes a subpopulation of rules with MoE."""

    input_size_: int
    complexity_: int

    def __init__(self, genome: np.ndarray, pool: list[Rule], mixing: MixingModel, fitness: IndividualFitness):
        self.genome = genome
        self.pool = pool
        self.mixing = mixing
        self.fitness = fitness

    def fit(self, X: np.ndarray, y: np.ndarray) -> Individual:
        pred = self.predict(X, cache=True)
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

    def clone(self, **kwargs) -> Individual:
        args = dict(
            genome=self.genome.copy() if 'genome' not in kwargs else None,
            pool=self.pool,
            mixing=self.mixing,
            fitness=self.fitness
        )
        return Individual(**(args | kwargs))

    def _more_str_attributes(self) -> dict:
        return {'complexity': self.complexity_}
