from abc import abstractmethod, ABCMeta
from typing import Callable

import numpy as np

from suprb2.individual import Individual
from suprb2.optimizer.fitness import emary, pseudo_accuracy, wu, BaseFitness


class IndividualFitness(BaseFitness, metaclass=ABCMeta):
    """Evaluate the fitness of a `Individual`."""

    @abstractmethod
    def __call__(self, individual: Individual) -> float:
        pass


class PseudoBIC(IndividualFitness):
    """Tries to minimize complexity along with error. Can be negative."""

    def __call__(self, individual: Individual) -> float:
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
                                  c_norm(individual.complexity_, individual.input_size_)) * 100


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
