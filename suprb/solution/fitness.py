from abc import ABCMeta
from typing import Callable

import numpy as np

from suprb.fitness import emary, pseudo_accuracy, wu
from . import Solution, SolutionFitness


class PseudoBIC(SolutionFitness):
    """Tries to minimize complexity along with error. Can be negative."""

    def __call__(self, solution: Solution) -> float:
        # note that error is capped to 1e-4 in suprb.solution.Solution.fit
        offset = 1 if solution.task == "Classification" else 0
        return -(solution.input_size_ * np.log(offset + solution.score_)
                 + solution.complexity_ * np.log(solution.input_size_))


def c_norm(complexity, N) -> float:
    return 1 - complexity / N


class ComplexitySolutionFitness(SolutionFitness, metaclass=ABCMeta):
    """Tries to minimize complexity along with error."""

    fitness_func_: Callable

    def __init__(self, alpha: float):
        self.alpha = alpha

    def __call__(self, solution: Solution) -> float:
        acc = -solution.score_ if solution.task == "Classification" else pseudo_accuracy(solution.score_)
        return self.fitness_func_(self.alpha, acc,
                                  c_norm(solution.complexity_, self.max_genome_length_)) * 100


class ComplexityEmary(ComplexitySolutionFitness):
    """
    Mixes the pseudo-accuracy as primary objective x1
    and c_norm as secondary objective x2 using the Emary fitness.
    Computed fitness is in [0, 100].
    """

    def __init__(self, alpha: float = 0.85):
        super().__init__(alpha=alpha)
        self.fitness_func_ = emary


class ComplexityWu(ComplexitySolutionFitness):
    """
    Mixes the pseudo-accuracy as primary objective x1
    and c_norm as secondary objective x2 using the Wu fitness.
    Computed fitness is in [0, 100].
    """

    def __init__(self, alpha: float = 0.3):
        super().__init__(alpha=alpha)
        self.fitness_func_ = wu
