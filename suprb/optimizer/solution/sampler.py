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


class UniformSolutionSampler(SolutionSampler):

    def __call__(self, pareto_front: list[Solution], random_state: RandomState) -> Solution:
        return random_state.choice(pareto_front)


class NormalSolutionSampler(SolutionSampler):

    def __call__(self, pareto_front: list[Solution], random_state: RandomState) -> Solution:
        weights = stats.norm.pdf(np.linspace(-1, 1, len(pareto_front)), 0, 1)
        weights = weights / np.sum(weights)
        return random_state.choice(pareto_front, p=weights)
