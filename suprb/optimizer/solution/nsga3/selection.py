from abc import ABCMeta

import numpy as np

from suprb.base import BaseComponent
from suprb.solution import Solution
from suprb.utils import RandomState


class SolutionSelection(BaseComponent, metaclass=ABCMeta):

    def __call__(
        self,
        population: list[Solution],
        n: int,
        random_state: RandomState,
        pareto_ranks: np.ndarray,
        closest_ref_direction: np.ndarray,
        ref_direction_distance: np.ndarray
    ) -> list[Solution]:
        pass


class ReferenceBasedBinaryTournament(SolutionSelection):
    """Draw 2 solutions n_parents times and select the best solution from each pair."""

    def __init__(self):
        pass

    def __call__(
        self,
        population: list[Solution],
        n: int,
        random_state: RandomState,
        pareto_ranks: np.ndarray,
        closest_ref_direction: np.ndarray,
        ref_direction_distance: np.ndarray
    ) -> list[Solution]:

        selection = []
        pairs = random_state.integers(low=0, high=len(population), size=(n, 2))
        for a, b in pairs:
            # select one solution randomly if they are not associated to the same reference direction
            if closest_ref_direction[a] != closest_ref_direction[b]:
                selection.append(population[a])
                continue
            # select the solution that is closer to the reference line if pareto ranks are equal
            if pareto_ranks[a] == pareto_ranks[b]:
                if ref_direction_distance[a] < ref_direction_distance[b]:
                    selection.append(population[a])
                    continue
                else:
                    selection.append(population[b])
                    continue
            # If pareto ranks are not equal select the less dominated solution
            if pareto_ranks[a] < pareto_ranks[b]:
                selection.append(population[a])
                continue
            else:
                selection.append(population[b])
                continue

        return selection
