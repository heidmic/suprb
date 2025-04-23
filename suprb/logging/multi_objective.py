import numpy as np

from . import DefaultLogger
from ..base import BaseRegressor

from .metrics import hypervolume, spread


class MOLogger(DefaultLogger):

    pareto_fronts_: dict

    def log_init(self, X: np.ndarray, y: np.ndarray, estimator: BaseRegressor):
        super().log_init(X, y, estimator)
        self.pareto_fronts_ = {}

    def log_iteration(self, X: np.ndarray, y: np.ndarray, estimator: BaseRegressor, iteration: int):
        super().log_iteration(X, y, estimator, estimator.step_)
        current_front = estimator.solution_composition_.pareto_front()
        if isinstance(current_front[0].fitness_, list):
            self.pareto_fronts_[iteration] = [solution.fitness_ for solution in current_front]
            # Very hacky way to check if we are in the first Stage of TS or not
            self.log_metric("hypervolume", hypervolume(current_front), estimator.step_)
            self.log_metric("spread", spread(current_front), estimator.step_)
