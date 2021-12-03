from abc import ABCMeta

import numpy as np


from suprb2.base import BaseComponent
from suprb2.individual import Individual


class IndividualMutation(BaseComponent, metaclass=ABCMeta):

    def __init__(self, mutation_rate: float = 0.05):
        self.mutation_rate = mutation_rate

    def __call__(self, individual: Individual, random_state: np.random.RandomState) -> Individual:
        pass


class BitFlips(IndividualMutation):
    """Flips every bit in the genome with probability `mutation_rate`."""

    def __call__(self, individual: Individual, random_state: np.random.RandomState) -> Individual:
        bit_flips = random_state.random(individual.genome.shape) < self.mutation_rate
        genome = np.logical_xor(individual.genome, bit_flips)
        return individual.clone(genome=genome)
