from abc import ABCMeta

import numpy as np

from suprb2.base import BaseComponent
from suprb2.individual import Individual


class IndividualCrossover(BaseComponent, metaclass=ABCMeta):

    def __call__(self, A: Individual, B: Individual, random_state: np.random.RandomState) -> Individual:
        pass


class NPoint(IndividualCrossover):
    """Cut the genome at N points and alternate the pieces from individual A and B."""

    def __init__(self, n: int = 2):
        self.n = n

    @staticmethod
    def _single_point(A: Individual, B: Individual, index: int) -> Individual:
        return Individual(genome=np.append(A.genome[:index], B.genome[index:]), pool=A.pool, mixture=A.mixture)

    def __call__(self, A: Individual, B: Individual, random_state: np.random.RandomState) -> Individual:
        indices = random_state.choice(np.arange(len(A.genome)), size=self.n, replace=False)
        for index in indices:
            A = self._single_point(A, B, index)
            B = self._single_point(B, A, index)
        return A


class Uniform(IndividualCrossover):
    """Decide for every bit with uniform probability if the bit in genome A or B is used."""

    def __call__(self, A: Individual, B: Individual, random_state: np.random.RandomState) -> Individual:
        indices = random_state.random(size=len(A.genome)) <= 0.5
        genome = np.empty(A.genome.shape)
        genome[indices] = A.genome[indices]
        genome[~indices] = B.genome[~indices]

        return Individual(genome=genome, pool=A.pool, mixture=A.mixture)
