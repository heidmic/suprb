from abc import ABCMeta

import numpy as np

from suprb2.base import BaseComponent
from suprb2.individual import Individual
from suprb2.utils import RandomState


class IndividualSelection(BaseComponent, metaclass=ABCMeta):

    def __call__(self, population: list[Individual], n: int, random_state: RandomState) -> list[Individual]:
        pass


class Random(IndividualSelection):
    """Sample `n_parents` at random."""

    def __call__(self, population: list[Individual], n: int, random_state: RandomState) -> list[Individual]:
        return list(random_state.choice(population, size=n))


class RouletteWheel(IndividualSelection):
    """Sample `n_parents` individuals proportional to their fitness."""

    def __call__(self, population: list[Individual], n: int, random_state: RandomState) -> list[Individual]:
        fitness_sum = sum([individual.fitness_ for individual in population])
        if fitness_sum != 0:
            weights = [individual.fitness_ / fitness_sum for individual in population]
        else:
            weights = None
        return list(random_state.choice(population, p=weights, size=n))


class LinearRank(IndividualSelection):
    """Sample `n_parents` individuals linear to their fitness ranking."""

    def __call__(self, population: list[Individual], n: int, random_state: RandomState) -> list[Individual]:
        fitness = np.array([individual.fitness_ for individual in population])
        ranks = fitness.argsort().argsort() + 1  # double `argsort()` obtains the ranks
        weights = ranks / sum(ranks)
        return list(random_state.choice(population, p=weights, size=n))


class Tournament(IndividualSelection):
    """Draw k individuals n_parents times and select the best individual from each k-subset."""

    def __init__(self, k: int = 5):
        self.k = k

    def __call__(self, population: list[Individual], n: int, random_state: RandomState) -> list[Individual]:
        return list(max(random_state.choice(population, size=self.k), key=lambda i: i.fitness_)
                    for _ in range(n))
