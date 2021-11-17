from abc import ABCMeta

import numpy as np

from suprb2.base import BaseComponent
from suprb2.individual import Individual


class IndividualSelection(BaseComponent, metaclass=ABCMeta):
    def __init__(self, parent_ratio: float = 0.2):
        self.parent_ratio = parent_ratio

    def n_parents(self, population_size: int) -> int:
        return int(population_size * self.parent_ratio)

    def __call__(self, population: list[Individual], random_state: np.random.RandomState) -> list[Individual]:
        pass


class Ranking(IndividualSelection):
    """Return the best `n_parents` individuals."""

    def __call__(self, population: list[Individual], random_state: np.random.RandomState) -> list[Individual]:
        return sorted(population, key=lambda i: i.fitness_, reverse=True)[:self.n_parents(len(population))]


class RouletteWheel(IndividualSelection):
    """Sample `n_parents` individuals proportional to their fitness."""

    def __call__(self, population: list[Individual], random_state: np.random.RandomState) -> list[Individual]:
        fitness_sum = sum([individual.fitness_ for individual in population])
        weights = [individual.fitness_ / fitness_sum for individual in population]
        return list(random_state.choice(population, p=weights, size=self.n_parents(len(population))))


class Tournament(IndividualSelection):
    """Draw k individuals n_parents times and select the best individual from each k-subset."""

    def __init__(self, parent_ratio: float = 0.2, k=5):
        super().__init__(parent_ratio=parent_ratio)
        self.k = k

    def __call__(self, population: list[Individual], random_state: np.random.RandomState) -> list[Individual]:
        return list(max(random_state.choice(population, size=self.k), key=lambda i: i.fitness_)
                    for _ in range(self.n_parents(len(population))))
