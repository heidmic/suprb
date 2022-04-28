from abc import ABCMeta
from typing import Optional

import numpy as np
from itertools import tee, permutations
from suprb import Rule, Solution
from suprb.base import BaseComponent
from suprb.utils import RandomState


class SolutionBuilder(BaseComponent, metaclass=ABCMeta):
    """Constructs the solutions by traversing a solution graph."""

    def __init__(self, alpha: float = 1, beta: float = 1, tau0: float = 0.5):
        self.alpha = alpha
        self.beta = beta
        self.tau0 = tau0

    def __call__(self, solution: Solution, pheromones: np.ndarray, pool: list[Rule],
                 random_state: RandomState) -> Solution:
        pass

    def update_pheromones(self, solution: Solution, pheromones: np.ndarray, delta_tau: float):
        """Apply the pheromone update to the pheromone matrix."""
        pass

    def pad_pheromone_matrix(self, pheromones: Optional[np.ndarray], size: int) -> np.ndarray:
        """Pad the pheromone matrix for additional rules."""
        pass


def relative_fitness(pool: list[Rule], scale=0.5) -> np.ndarray:
    """Calculates the relative fitness of every rule, i.e., normalizes the fitness and scale to [1-scale, 1+scale]."""

    assert scale < 1

    fitness = np.array([rule.fitness_ for rule in pool])
    normalized = ((fitness - np.amin(fitness)) / (np.amax(fitness) - np.amin(fitness))) * scale

    return 1 + np.stack((-normalized, normalized), axis=1)


class Binary(SolutionBuilder):
    """
    Represents a binary solution graph, i.e., traverse the rules by their insertion order.

    Taken from https://doi.org/10/dnxz32.
    """

    def __init__(self, alpha: float = 1, beta: float = 1, tau0: float = 5):
        super().__init__(alpha, beta, tau0)

    def pad_pheromone_matrix(self, pheromones: Optional[np.ndarray], size: int) -> np.ndarray:
        """Initialize and pad a Nx2 pheromone matrix."""
        if pheromones is None:
            return self.pad_pheromone_matrix(np.empty((0, 2), dtype='float64'), size)
        else:
            return np.pad(pheromones, ((0, size - pheromones.shape[0]), (0, 0)), mode='constant',
                          constant_values=self.tau0)

    def __call__(self, solution: Solution, pheromones: np.ndarray, pool: list[Rule],
                 random_state: RandomState) -> Solution:
        eta = relative_fitness(pool)

        # Calculate weights and normalize
        weights = pheromones ** self.alpha * eta ** self.beta
        weights = weights / np.sum(weights, axis=1).reshape((-1, 1))

        route = random_state.random(size=len(pool)) <= weights[:, 1]
        solution.genome = route

        return solution

    def update_pheromones(self, solution: Solution, pheromones: np.ndarray, delta_tau: float):
        # Add the delta tau to all rules (de)selected
        pheromones[:, 0] += ~solution.genome * delta_tau
        pheromones[:, 1] += solution.genome * delta_tau


def pairwise(iterable):
    """Sliding window of size two with circular end."""
    a, b = tee(iterable + [iterable[0]])
    next(b, None)
    return zip(a, b)


def relative_bounds_overlap(A: Rule, B: Rule) -> np.array:
    """
    Calculate the relative amount of how much the bounds of two rules overlap.
    If one rule lies completely within the bounds of the other rule, the overlap is 1.
    """

    intersections = (A.bounds[:, 0] <= B.bounds[:, 1]) & (B.bounds[:, 0] <= A.bounds[:, 1])

    if not np.all(intersections):
        return np.ones(2)

    ones = np.ones(A.bounds.shape[0])
    ones[intersections] = np.min(np.stack((A.bounds[:, 1], B.bounds[:, 1]), axis=1), axis=1) - np.max(
        np.stack((A.bounds[:, 0], B.bounds[:, 0]), axis=1), axis=1)
    overlap = np.prod(ones)
    shared_relative_volume = overlap / (min(A.volume_, B.volume_))

    return 1 + np.array([shared_relative_volume, -shared_relative_volume])


class Complete(SolutionBuilder):
    """
    Represents a complete solution graph. Two modes of operation are provided to select the next rule:
    - use_partial_route = True: all rules that are currently selected are considered.
    - use_partial_route = False: only the last selected rule is considered.

    Inspired by https://doi.org/10/gnfbnv.
    """

    def __init__(self, alpha: float = 1, beta: float = 1, tau0: float = 5, use_partial_route: bool = True):
        super().__init__(alpha, beta, tau0)

        self.use_partial_route = use_partial_route

    def pad_pheromone_matrix(self, pheromones: Optional[np.ndarray], size: int):
        """Initialize and pad a NxNx2 pheromone matrix."""
        if pheromones is None:
            return self.pad_pheromone_matrix(np.empty((0, 0, 2), dtype='float64'), size)
        else:
            padding = size - pheromones.shape[0]
            return np.pad(pheromones, ((0, padding), (0, padding), (0, 0)), mode='constant', constant_values=self.tau0)

    def __call__(self, solution: Solution, pheromones: np.ndarray, pool: list[Rule],
                 random_state: RandomState) -> Solution:

        # Initialize the route and relative fitness values
        route = []
        fitness = relative_fitness(pool)

        # Traverse the rules in random order
        order = list(range(len(pool)))
        random_state.shuffle(order)

        for j in order:
            # Start with no overlap and all pheromones, if no rule was selected yet
            if not route:
                overlap = np.ones(2)
                tau = np.sum(pheromones[:, j, :], axis=0)
            else:
                # Calculate pheromones and overlap, depending on the mode of operation
                if self.use_partial_route:
                    overlap = np.mean(np.stack([relative_bounds_overlap(pool[i], pool[j]) for i in route], axis=0),
                                      axis=0)
                    tau = np.sum(pheromones[route + [j], j, :], axis=0)
                else:
                    overlap = relative_bounds_overlap(pool[route[-1]], pool[j])
                    tau = np.sum(pheromones[[route[-1], j], j, :], axis=0)
            # Combine overlap and fitness to get the heuristic values
            eta = np.sqrt(fitness[j] * overlap)
            # Combine pheromones and heuristic values and normalize the weights
            weights = tau ** self.alpha * eta ** self.beta
            weights = weights / np.sum(weights)

            # Decide if the current rule should be selected
            if random_state.random() <= weights[1]:
                route.append(j)

        # Encode the selected rules
        solution.genome = np.zeros(len(pool), dtype='bool')
        solution.genome[route] = 1

        return solution

    def update_pheromones(self, solution: Solution, pheromones: np.ndarray, delta_tau: float):
        # Update the pheromones of all rule pair permutations of selected rules
        selected_indices = np.nonzero(solution.genome)[0]
        if len(selected_indices) > 1:
            selected_permutations = np.array(list(permutations(selected_indices, 2)))
            pheromones[selected_permutations[:, 0], selected_permutations[:, 1], 1] += delta_tau

        pheromones[selected_indices, selected_indices, 1] += delta_tau

        deselected_indices = np.nonzero(~solution.genome)[0]
        pheromones[deselected_indices, deselected_indices, 0] += delta_tau

        # Update the pheromones of all (deselected, selected) pairs and vice versa
        for i in selected_indices:
            for j in deselected_indices:
                pheromones[i, j, 0] += delta_tau
                pheromones[j, i, 1] += delta_tau
