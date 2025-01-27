from abc import ABCMeta, abstractmethod

import numpy as np

from suprb.base import BaseComponent
from suprb.solution import Solution
from suprb.utils import RandomState


class SolutionCrossover(BaseComponent, metaclass=ABCMeta):

    def __init__(self):
        pass

    def __call__(
        self,
        A: Solution,
        B: Solution,
        crossover_rate_min,
        crossover_rate_max,
        fitness_mean,
        fitness_min,
        fitness_max,
        random_state: RandomState,
    ) -> Solution:
        fitness_parents_mean = (A.fitness_ + B.fitness_) / 2
        if fitness_min == fitness_mean or fitness_mean == fitness_max:
            crossover_rate = crossover_rate_max
        elif fitness_parents_mean <= fitness_mean:
            crossover_rate = crossover_rate_max - (crossover_rate_max - crossover_rate_min) * (
                (fitness_mean - fitness_parents_mean) / (fitness_mean - fitness_min)
            )
        else:
            crossover_rate = crossover_rate_max - (crossover_rate_max - crossover_rate_min) * (
                (fitness_parents_mean - fitness_mean) / (fitness_max - fitness_mean)
            )

        if random_state.random() < crossover_rate:
            return self._crossover(A=A, B=B, random_state=random_state)
        else:
            # Just return the primary parent
            return A

    @abstractmethod
    def _crossover(self, A: Solution, B: Solution, random_state: RandomState) -> Solution:
        pass


class NPoint(SolutionCrossover):
    """Cut the genome at N points and alternate the pieces from solution A and B."""

    def __init__(self, n: int = 2):
        super().__init__()
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
