import numpy as np

from . import DefaultLogger
from ..base import BaseRegressor

from .metrics import hypervolume, spread
from ..optimizer.solution.base import MOSolutionComposition, SolutionComposition
from ..optimizer.solution.ts import TwoStageSolutionComposition


def get_active_sc(es: BaseRegressor):
    sc = es.solution_composition_
    if isinstance(sc, TwoStageSolutionComposition):
        return sc.current_algo_
    else:
        return sc


class MOLogger(DefaultLogger):

    pareto_fronts_: dict

    def log_init(self, X: np.ndarray, y: np.ndarray, estimator: BaseRegressor):
        super().log_init(X, y, estimator)
        self.pareto_fronts_ = {}

    def log_iteration(self, X: np.ndarray, y: np.ndarray, estimator: BaseRegressor, iteration: int):
        super().log_iteration(X, y, estimator, estimator.step_)
        sc = get_active_sc(estimator)
        if isinstance(sc, MOSolutionComposition):
            current_front = sc.pareto_front()
            self.pareto_fronts_[iteration] = [solution.fitness_ for solution in current_front]
            self.log_metric("hypervolume", hypervolume(current_front), estimator.step_)
            self.log_metric("spread", spread(current_front), estimator.step_)
            self.log_metric("sc_iterations", sc.step_, estimator.step_)
