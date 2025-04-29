from abc import ABCMeta, abstractmethod

import scipy.stats as stats

from suprb.base import BaseComponent
from suprb.solution import Solution
from suprb.utils import RandomState

import numpy as np


def calculate_crowding_distances(fitness_values: np.ndarray) -> np.ndarray:
    objective_count = fitness_values.shape[1]
    crowding_distances = np.zeros(fitness_values.shape[0])

    if fitness_values.shape[0] <= 2:
        crowding_distances[:] = 1

    for m in range(objective_count):
        sorting_permutation = np.argsort(fitness_values[:, m])
        sorted_front = fitness_values[sorting_permutation]

        crowding_distances[sorted_front[0]] = np.inf
        crowding_distances[sorted_front[-1]] = np.inf

        min_f = fitness_values[sorting_permutation[0], m]
        max_f = fitness_values[sorting_permutation[-1], m]

        # if max_f == min_f the crowding distance parts that result from objective_m are all 0 as all
        # solution share the same coordinate in this dimension of the fitness function
        if max_f > min_f:
            normalized_range = max_f - min_f
            for i in range(1, len(crowding_distances) - 1):
                crowding_distances[sorted_front[i]] += (
                    fitness_values[sorting_permutation[i + 1], m] - fitness_values[sorting_permutation[i - 1], m]
                ) / normalized_range

    return crowding_distances


class SolutionSampler(BaseComponent, metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, pareto_front: list[Solution], random_state: RandomState) -> Solution:
        pass


class NormalSolutionSampler(SolutionSampler):

    def __init__(self, sigma: float = 0.2):
        self.sigma = sigma

    def __call__(self, pareto_front: list[Solution], random_state: RandomState) -> Solution:
        weights = stats.norm.pdf(np.linspace(0, 1, len(pareto_front)), 0.5, self.sigma)
        weights = weights / np.sum(weights)
        return random_state.choice(pareto_front, p=weights)


class BetaSolutionSampler(SolutionSampler):

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, projected: bool = True):
        self.alpha = alpha
        self.beta = beta
        self.projected = projected

    def __call__(self, pareto_front: list[Solution], random_state: RandomState) -> Solution:
        if self.projected:
            points = np.array([solution.fitness_ for solution in pareto_front])
            points = points / np.sum(points, axis=1)
            points = points[:, 0]
        else:
            points = np.linspace(0, 1, len(pareto_front))

        weights = stats.norm.beta(points, self.alpha, self.beta)
        weights = weights / np.sum(weights)
        return random_state.choice(pareto_front, p=weights)


class DiversitySolutionSampler(SolutionSampler):

    def __call__(self, pareto_front: list[Solution], random_state: RandomState) -> Solution:
        fitness_values = np.array([solution.fitness_ for solution in pareto_front])
        weights = calculate_crowding_distances(fitness_values)
        weights = weights / np.sum(weights)
        return random_state.choice(pareto_front, p=weights)
