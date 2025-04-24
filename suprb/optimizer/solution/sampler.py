from abc import ABCMeta, abstractmethod

import scipy.stats as stats

from suprb.base import BaseComponent
from suprb.solution import Solution
from suprb.utils import RandomState

import numpy as np


class SolutionSampler(BaseComponent, metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, pareto_front: list[Solution], random_state: RandomState) -> Solution:
        pass


class NormalSolutionSampler(SolutionSampler):

    def __init__(self, mu: float = 0.0):
        self.mu = mu

    def __call__(self, pareto_front: list[Solution], random_state: RandomState) -> Solution:
        weights = stats.norm.pdf(np.linspace(0, 1, len(pareto_front)), 0.5, self.mu)
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
