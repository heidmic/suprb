from abc import ABCMeta

import numpy as np

from suprb.base import BaseComponent
from suprb.solution import Solution
from suprb.utils import RandomState


class SolutionSelection(BaseComponent, metaclass=ABCMeta):

    def __call__(self, population: list[Solution], n: int, random_state: RandomState) -> list[Solution]:
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
        pareto_ranks: np.ndarray,
        crowding_distances: np.ndarray
    ) -> list[Solution]:

        selection = []
        pairs = random_state.integers(low=0, high=len(population), size=(n, 2))
        for a, b in pairs:
            if pareto_ranks[a] == pareto_ranks[b]:
                if crowding_distances[a] >= crowding_distances[b]:
                    selection.append(population[a])
                else:
                    selection.append(population[b])
            elif pareto_ranks[a] > pareto_ranks[b]:
                selection.append(population[a])
            else:
                selection.append(population[b])
        return selection
