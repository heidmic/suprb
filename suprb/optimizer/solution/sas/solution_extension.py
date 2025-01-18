from __future__ import annotations

import itertools
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error
from suprb.base import BaseComponent
from suprb.optimizer.solution.archive import Elitist

from suprb.rule import Rule
from suprb.solution.base import MixingModel, Solution, SolutionFitness
from suprb.solution.initialization import RandomInit
from suprb.utils import RandomState


def padding_size(solution: Solution) -> int:
    """Calculates the number of bits to add to the genome after the pool was expanded."""

    return len(solution.pool) - solution.genome.shape[0]


def random(n: int, p: float, random_state: RandomState):
    """Returns a random bit string of size `n`, with ones having probability `p`."""

    return (random_state.random(size=n) <= p).astype("bool")


class SasSolution(Solution):
    """Solution that mixes a subpopulation of rules. Extended to have an age"""

    def __init__(
        self,
        genome: np.ndarray,
        pool: list[Rule],
        mixing: MixingModel,
        fitness: SolutionFitness,
        age: int = 3,
    ):
        super().__init__(genome, pool, mixing, fitness)
        self.age = age

    def fit(self, X: np.ndarray, y: np.ndarray) -> SasSolution:
        pred = self.predict(X, cache=True)
        self.error_ = max(mean_squared_error(y, pred), 1e-4)
        self.input_size_ = self.genome.shape[0]
        self.complexity_ = np.sum(
            self.genome
        ).item()  # equivalent to np.count_nonzero, but possibly faster
        self.fitness_ = self.fitness(self)
        self.is_fitted_ = True
        return self

    def clone(self, **kwargs) -> SasSolution:
        args = dict(
            genome=self.genome.copy() if "genome" not in kwargs else None,
            pool=self.pool,
            mixing=self.mixing,
            fitness=self.fitness,
        )
        solution = SasSolution(**(args | kwargs))
        if not kwargs:
            attributes = [
                "fitness_",
                "error_",
                "complexity_",
                "is_fitted_",
                "input_size_",
            ]
            solution.__dict__ |= {key: getattr(self, key) for key in attributes}
        return solution
