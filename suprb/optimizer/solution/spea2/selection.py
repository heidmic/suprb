from abc import ABCMeta

import numpy as np

from suprb.base import BaseComponent
from suprb.solution import Solution
from suprb.utils import RandomState


class SolutionSelection(BaseComponent, metaclass=ABCMeta):

    def __call__(
        self, population: list[Solution], n: int, random_state: RandomState, internal_fitness: np.ndarray
    ) -> list[Solution]:
        pass


class BinaryTournament(SolutionSelection):
    """Draw 2 solutions n_parents times and select the best solution from each pair."""

    def __init__(self):
        pass

    def __call__(
        self,
        population: list[Solution],
        n: int,
        random_state: RandomState,
        internal_fitness: np.ndarray,
    ) -> list[Solution]:

        selection = []
        pairs = random_state.integers(low=0, high=len(population), size=(n, 2))
        for a, b in pairs:
            if internal_fitness[a] < internal_fitness[b]:
                selection.append(population[a])
            else:
                selection.append(population[b])
        return selection
