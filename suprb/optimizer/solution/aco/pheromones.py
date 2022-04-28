from abc import ABCMeta

from suprb import Solution
from suprb.base import BaseComponent


class PheromoneUpdate(BaseComponent, metaclass=ABCMeta):

    def __init__(self, c: float = 0.01):
        self.c = c

    def __call__(self, solution: Solution) -> float:
        pass


class Fitness(PheromoneUpdate):

    def __call__(self, solution: Solution) -> float:
        return solution.fitness_ * self.c
