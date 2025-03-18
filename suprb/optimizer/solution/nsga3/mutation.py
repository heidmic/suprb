from abc import ABCMeta

import numpy as np

from suprb.base import BaseComponent
from suprb.solution import Solution
from suprb.utils import RandomState


class SolutionMutation(BaseComponent, metaclass=ABCMeta):

    def __init__(self, mutation_rate: float = 0.05):
        self.mutation_rate = mutation_rate

    def __call__(self, solution: Solution, random_state: RandomState) -> Solution:
        pass


class BitFlips(SolutionMutation):
    """Flips every bit in the genome with probability `mutation_rate`."""

    def __call__(self, solution: Solution, random_state: RandomState) -> Solution:
        bit_flips = random_state.random(solution.genome.shape) < self.mutation_rate
        genome = np.logical_xor(solution.genome, bit_flips)
        return solution.clone(genome=genome)
