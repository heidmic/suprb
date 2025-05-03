from abc import ABCMeta, abstractmethod
import scipy.stats as stats
from typing import Optional, Callable
from sklearn.utils import Bunch

from suprb.base import BaseComponent
from suprb.solution import Solution
from suprb.utils import RandomState

import numpy as np


def calculate_crowding_distances(fitness_values: np.ndarray, pareto_ranks: np.ndarray) -> np.ndarray:
    solution_count, objective_count = fitness_values.shape
    crowding_distances = np.zeros(solution_count)

    for rank in np.unique(pareto_ranks):
        front_indices = np.where(pareto_ranks == rank)[0]
        front_size = len(front_indices)

        if front_size <= 2:
            crowding_distances[front_indices] = np.inf
            continue

        front_fitness = fitness_values[front_indices]

        for m in range(objective_count):
            sorting_permutation = np.argsort(front_fitness[:, m])
            sorted_front = front_indices[sorting_permutation]

            crowding_distances[sorted_front[0]] = np.inf
            crowding_distances[sorted_front[-1]] = np.inf

            min_f = front_fitness[sorting_permutation[0], m]
            max_f = front_fitness[sorting_permutation[-1], m]

            # if max_f == min_f the crowding distance parts that result from objective_m are all 0 as all
            # solution share the same coordinate in this dimension of the fitness function
            if max_f > min_f:
                normalized_range = max_f - min_f
                for i in range(1, front_size - 1):
                    crowding_distances[sorted_front[i]] += (
                        front_fitness[sorting_permutation[i + 1], m] - front_fitness[sorting_permutation[i - 1], m]
                    ) / normalized_range

    return crowding_distances


class SolutionSampler(BaseComponent, metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, pareto_front: list[Solution], random_state: RandomState) -> Solution:
        pass


class BetaSolutionSampler(SolutionSampler):

    def __init__(self, a: float = 1.5, b: float = 1.5, projected: bool = True):
        self.a = a
        self.b = b
        self.projected = projected

    def __call__(self, pareto_front: list[Solution], random_state: RandomState) -> Solution:
        if self.projected:
            points = np.array([solution.fitness_ for solution in pareto_front])
            points = points / np.sum(points, axis=1, keepdims=True)
            points = points[:, 0]
        else:
            points = np.linspace(0.0001, 1 - 0.0001, len(pareto_front))

        weights = stats.beta.pdf(points, a=self.a, b=self.b)
        weights = weights / np.sum(weights)
        return random_state.choice(pareto_front, p=weights)


class DiversitySolutionSampler(SolutionSampler):

    def __call__(self, pareto_front: list[Solution], random_state: RandomState) -> Solution:
        fitness_values = np.array([solution.fitness_ for solution in pareto_front])
        weights = calculate_crowding_distances(fitness_values, np.zeros(len(pareto_front)))
        weights[weights == np.inf] = 0
        weights = (weights / np.sum(weights)) if np.sum(weights) > 0 else np.ones(len(weights)) / len(weights)
        return random_state.choice(pareto_front, p=weights)
