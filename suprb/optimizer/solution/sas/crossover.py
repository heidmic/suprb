from abc import ABCMeta, abstractmethod

import numpy as np

from suprb.base import BaseComponent
from .solution_extension import SasSolution
from suprb.solution import Solution
from suprb.utils import RandomState


class SolutionCrossover(BaseComponent, metaclass=ABCMeta):

    def __init__(self, crossover_rate: float = 0.9):
        self.crossover_rate = crossover_rate

    def __call__(self, A: SasSolution, B: SasSolution, random_state: RandomState) -> SasSolution:
        if random_state.random() < self.crossover_rate:
            return self._crossover(A=A, B=B, random_state=random_state)
        else:
            # Just return the primary parent
            return A

    @abstractmethod
    def _crossover(self, A: SasSolution, B: SasSolution, random_state: RandomState) -> SasSolution:
        pass


class NPoint(SolutionCrossover):
    """Cut the genome at N points and alternate the pieces from solution A and B."""

    def __init__(self, crossover_rate: float = 0.9, n: int = 2):
        super().__init__(crossover_rate=crossover_rate)
        self.n = n

    @staticmethod
    def _single_point(A: SasSolution, B: SasSolution, index: int) -> SasSolution:
        return A.clone(genome=np.append(A.genome[:index], B.genome[index:]))

    def _crossover(self, A: SasSolution, B: SasSolution, random_state: RandomState) -> SasSolution:
        indices = random_state.choice(np.arange(len(A.genome)), size=min(self.n, len(A.genome)), replace=False)
        for index in indices:
            A = self._single_point(A, B, index)
            B = self._single_point(B, A, index)
        return A


class Uniform(SolutionCrossover):
    """Decide for every bit with uniform probability if the bit in genome A or B is used."""

    def _crossover(self, A: SasSolution, B: SasSolution, random_state: RandomState) -> SasSolution:
        indices = random_state.random(size=len(A.genome)) <= 0.5
        genome = np.empty(A.genome.shape)
        genome[indices] = A.genome[indices]
        genome[~indices] = B.genome[~indices]

        return A.clone(genome=genome)
