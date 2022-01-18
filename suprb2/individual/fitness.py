from abc import ABCMeta
from typing import Callable

import numpy as np

from suprb2.fitness import emary, pseudo_accuracy, wu
from . import Individual, IndividualFitness


class PseudoBIC(IndividualFitness):
    """Tries to minimize complexity along with error. Can be negative."""

    def __call__(self, individual: Individual) -> float:
        # note that error is capped to 1e-4 in suprb2.individual.Individual.fit
        return -(individual.input_size_ * np.log(individual.error_)
                 + individual.complexity_ * np.log(individual.input_size_))


def c_norm(complexity, N) -> float:
    return 1 - complexity / N


class ComplexityIndividualFitness(IndividualFitness, metaclass=ABCMeta):
    """Tries to minimize complexity along with error."""

    fitness_func_: Callable

    def __init__(self, alpha: float):
        self.alpha = alpha

    def __call__(self, individual: Individual) -> float:
        return self.fitness_func_(self.alpha, pseudo_accuracy(individual.error_),
                                  c_norm(individual.complexity_, self.max_genome_length_)) * 100


class ComplexityEmary(ComplexityIndividualFitness):
    """
    Mixes the pseudo-accuracy as primary objective x1
    and c_norm as secondary objective x2 using the Emary fitness.
    Computed fitness is in [0, 100].
    """

    def __init__(self, alpha: float = 0.85):
        super().__init__(alpha=alpha)
        self.fitness_func = emary


class ComplexityWu(ComplexityIndividualFitness):
    """
    Mixes the pseudo-accuracy as primary objective x1
    and c_norm as secondary objective x2 using the Wu fitness.
    Computed fitness is in [0, 100].
    """

    def __init__(self, alpha: float = 0.3):
        super().__init__(alpha=alpha)
        self.fitness_func_ = wu
