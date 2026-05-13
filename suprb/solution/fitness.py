from abc import ABCMeta
from typing import Callable, Tuple

import numpy as np

from suprb.fitness import emary, pseudo_accuracy, wu
from . import Solution, SolutionFitness


class PseudoBIC(SolutionFitness):
    """Tries to minimize complexity along with error. Can be negative."""

    def __call__(self, solution: Solution) -> float:
        # note that error is capped to 1e-4 in suprb.solution.Solution.fit
        return -(solution.input_size_ * np.log(solution.error_) + solution.complexity_ * np.log(solution.input_size_))


def c_norm(complexity, N) -> float:
    return 1 - complexity / N


class ComplexitySolutionFitness(SolutionFitness, metaclass=ABCMeta):
    """Tries to minimize complexity along with error."""

    fitness_func_: Callable

    def __init__(self, alpha: float):
        self.alpha = alpha

    def __call__(self, solution: Solution) -> float:
        return (
            self.fitness_func_(
                self.alpha,
                pseudo_accuracy(solution.error_),
                c_norm(solution.complexity_, self.max_genome_length_),
            )
            * 100
        )


class MultiObjectiveSolutionFitness(SolutionFitness, metaclass=ABCMeta):
    """Passes on fitness and complexity measures for MOOAs to minimize complexity along with error."""

    objective_func_: list[Callable]
    hv_reference_: np.ndarray
    worst_point_estimate_: np.ndarray

    def __init__(self):
        pass

    def __call__(self, solution: Solution) -> list:
        return [self.objective_func_[i](solution) for i in range(len(self.objective_func_))]


class NormalizedMOSolutionFitness(MultiObjectiveSolutionFitness):
    objective_func_: list[Callable] = [c_norm, pseudo_accuracy]
    hv_reference_: np.ndarray = np.array([1.0, 1.0])
    worst_point_estimate_ = np.array([1.0, 1.0])

    def __call__(self, solution: Solution) -> list:
        return [
            1 - self.objective_func_[0](solution.complexity_, self.max_genome_length_),
            1 - self.objective_func_[1](solution.error_),
        ]

    def __init__(self):
        super().__init__()


class SimpleMOSolutionFitness(MultiObjectiveSolutionFitness):

    def __call__(self, solution: Solution):
        return [solution.complexity_, solution.error_]


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
