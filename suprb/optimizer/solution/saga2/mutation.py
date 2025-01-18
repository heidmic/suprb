from abc import ABCMeta

import numpy as np


from suprb.base import BaseComponent
from suprb.solution import Solution
from suprb.utils import RandomState


class SolutionMutation(BaseComponent, metaclass=ABCMeta):

    def __init__(self):
        pass

    def __call__(
        self,
        solution: Solution,
        mutation_rate_min: float,
        mutation_rate_max: float,
        fitness_mean: float,
        fitness_min: float,
        fitness_max: float,
        random_state: RandomState,
    ) -> Solution:
        pass


class BitFlips(SolutionMutation):
    """Flips every bit in the genome with probability `mutation_rate`."""

    def __call__(
        self,
        solution: Solution,
        mutation_rate_min: float,
        mutation_rate_max: float,
        fitness_mean: float,
        fitness_max: float,
        random_state: RandomState,
    ) -> Solution:
        if fitness_max == fitness_mean:
            mutation_rate = mutation_rate_min
        elif solution.fitness_ > fitness_mean:
            mutation_rate = mutation_rate_min + (
                mutation_rate_max - mutation_rate_min
            ) * ((fitness_max - solution.fitness_) / (fitness_max - fitness_mean))
        else:
            mutation_rate = mutation_rate_max

        bit_flips = random_state.random(solution.genome.shape) < mutation_rate
        genome = np.logical_xor(solution.genome, bit_flips)
        return solution.clone(genome=genome)
