from abc import ABCMeta, abstractmethod

import numpy as np

from suprb2.base import BaseComponent
from suprb2.individual import Individual, MixtureOfExperts
from suprb2.rule import Rule


def padding_size(rule: Individual) -> int:
    """Calculates the number of bits to add to the genome after the pool was expanded."""

    return len(rule.pool) - rule.genome.shape[0]


def random(n: int, p: float, random_state: np.random.RandomState):
    """Returns a random bit string of size `n`, with ones having probability `p`."""

    return (random_state.random(size=n) <= p).astype('bool')


class IndividualInit(BaseComponent, metaclass=ABCMeta):
    """Generates initial genomes and pads existing genomes."""

    def __init__(self, mixture: MixtureOfExperts):
        self.mixture = mixture

    @abstractmethod
    def __call__(self, pool: list[Rule], random_state: np.random.RandomState) -> Individual:
        pass

    @abstractmethod
    def pad(self, individual: Individual, random_state: np.random.RandomState) -> Individual:
        pass


class ZeroInit(IndividualInit):
    """Init and extend genomes with zeros."""

    def __call__(self, pool: list[Rule], random_state: np.random.RandomState) -> Individual:
        return Individual(genome=np.zeros(len(pool), dtype='bool'), pool=pool, mixture=self.mixture)

    def pad(self, individual: Individual, random_state: np.random.RandomState = None) -> Individual:
        individual.genome = np.pad(individual.genome, (0, padding_size(individual)), mode='constant')
        return individual


class RandomInit(IndividualInit):
    """Init and extend genomes with random values, with `p` denoting the probability of ones."""

    def __init__(self, mixture: MixtureOfExperts, p: float = 0.5):
        super().__init__(mixture)
        self.p = p

    def __call__(self, pool: list[Rule], random_state: np.random.RandomState) -> Individual:
        return Individual(genome=random(len(pool), self.p, random_state), pool=pool, mixture=self.mixture)

    def pad(self, individual: Individual, random_state: np.random.RandomState) -> Individual:
        individual.genome = np.concatenate((individual.genome, random(padding_size(individual), self.p, random_state)),
                                           axis=0)
        return individual
