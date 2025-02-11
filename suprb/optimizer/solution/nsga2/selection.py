from abc import ABCMeta

import numpy as np

from suprb.base import BaseComponent
from suprb.solution import Solution
from suprb.utils import RandomState


class SolutionSelection(BaseComponent, metaclass=ABCMeta):

    def __call__(self, population: list[Solution], n: int, random_state: RandomState) -> list[Solution]:
        pass


class Random(SolutionSelection):
    """Sample `n_parents` at random."""

    def __call__(self, population: list[Solution], n: int, random_state: RandomState) -> list[Solution]:
        return list(random_state.choice(population, size=n))


class RouletteWheel(SolutionSelection):
    """Sample `n_parents` solutions proportional to their fitness."""

    def __call__(self, population: list[Solution], n: int, random_state: RandomState) -> list[Solution]:
        fitness_sum = sum([solution.fitness_ for solution in population])
        if fitness_sum != 0:
            weights = [solution.fitness_ / fitness_sum for solution in population]
        else:
            weights = None
        return list(random_state.choice(population, p=weights, size=n))


class LinearRank(SolutionSelection):
    """Sample `n_parents` solutions linear to their fitness ranking."""

    def __call__(self, population: list[Solution], n: int, random_state: RandomState) -> list[Solution]:
        fitness = np.array([solution.fitness_ for solution in population])
        ranks = fitness.argsort().argsort() + 1  # double `argsort()` obtains the ranks
        weights = ranks / sum(ranks)
        return list(random_state.choice(population, p=weights, size=n))


class Tournament(SolutionSelection):
    """Draw k solutions n_parents times and select the best solution from each k-subset."""

    def __init__(self, k: int = 5):
        self.k = k

    def __call__(self, population: list[Solution], n: int, random_state: RandomState) -> list[Solution]:
        return list(max(random_state.choice(population, size=self.k), key=lambda i: i.fitness_) for _ in range(n))
