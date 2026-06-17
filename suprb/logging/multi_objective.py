import numpy as np

from . import DefaultLogger
from ..base import BaseRegressor

from suprb.solution.fitness import c_norm, pseudo_accuracy

from .metrics import hypervolume, spread
from ..optimizer.solution.base import MOSolutionComposition, SolutionComposition
from ..optimizer.solution.ts import TwoStageSolutionComposition


class MOLogger(DefaultLogger):

    pareto_fronts_: dict

    def log_init(self, X: np.ndarray, y: np.ndarray, estimator: BaseRegressor):
        super().log_init(X, y, estimator)
        self.pareto_fronts_ = {}

    def log_iteration(self, X: np.ndarray, y: np.ndarray, estimator: BaseRegressor, iteration: int):
        super().log_iteration(X, y, estimator, estimator.step_)
        sc = estimator.solution_composition_
        current_front = sc.pareto_front()
        n = current_front[0].fitness.max_genome_length_
        if hasattr(current_front[0].fitness, "hv_reference"):
            reference_points = current_front[0].fitness.hv_reference
        else:
            reference_points = np.array([1.0, 1.0])
        current_front = [
            [1 - c_norm(solution.complexity_, n), 1 - pseudo_accuracy(solution.error_)] for solution in current_front
        ]
        self.pareto_fronts_[iteration] = current_front
        current_front = np.array(current_front)
        self.log_metric("hypervolume", hypervolume(current_front, reference_points), estimator.step_)
        self.log_metric("spread", spread(current_front), estimator.step_)
        self.log_metric("sc_iterations", sc.step_, estimator.step_)
