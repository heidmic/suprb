from abc import ABCMeta, abstractmethod

import numpy as np

from suprb.base import BaseComponent
from suprb.solution import Solution
from suprb.utils import RandomState


class SolutionCrossover(BaseComponent, metaclass=ABCMeta):
    def __call__(self, A: Solution, B: Solution, crossover_rate, random_state: RandomState) -> Solution:
        result = None

        if random_state.random() < crossover_rate:
            result = self._crossover(A=A, B=B, random_state=random_state)
        else:
            result = A

        return result

    @abstractmethod
    def _crossover(self, A: Solution, B: Solution, random_state: RandomState) -> Solution:
        pass


class NPoint(SolutionCrossover):
    """Cut the genome at N points and alternate the pieces from solution A and B."""

    def __init__(self, n: int = 2):
        self.n = n

    @staticmethod
    def _single_point(A: Solution, B: Solution, index: int) -> Solution:
        return A.clone(genome=np.append(A.genome[:index], B.genome[index:]))

    def _crossover(self, A: Solution, B: Solution, random_state: RandomState) -> Solution:
        indices = random_state.choice(np.arange(len(A.genome)), size=min(self.n, len(A.genome)), replace=False)
        for index in indices:
            A = self._single_point(A, B, index)

        return A


class Uniform(SolutionCrossover):
    """Decide for every bit with uniform probability if the bit in genome A or B is used."""

    def _crossover(self, A: Solution, B: Solution, random_state: RandomState) -> Solution:
        indices = random_state.random(size=len(A.genome)) <= 0.5
        genome = np.empty(A.genome.shape)
        genome[indices] = A.genome[indices]
        genome[~indices] = B.genome[~indices]

        return A.clone(genome=genome)
