from abc import ABCMeta

import numpy as np


from suprb.base import BaseComponent
from .solution_extension import SagaSolution
from suprb.utils import RandomState


class SolutionMutation(BaseComponent, metaclass=ABCMeta):

    def __init__(self):
        pass

    def __call__(
        self, solution: SagaSolution, random_state: RandomState
    ) -> SagaSolution:
        pass


class BitFlips(SolutionMutation):
    """Flips every bit in the genome with probability `mutation_rate`."""

    def __call__(
        self, solution: SagaSolution, random_state: RandomState
    ) -> SagaSolution:
        bit_flips = random_state.random(solution.genome.shape) < solution.mutation_rate
        genome = np.logical_xor(solution.genome, bit_flips)
        return solution.clone(genome=genome)
