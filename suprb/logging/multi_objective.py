import numpy as np

from . import DefaultLogger
from ..base import BaseRegressor
from suprb.solution import Solution

from .metrics import hypervolume, spread


class MOLogger(DefaultLogger):

    pareto_fronts_: list[list[Solution]] = []

    def log_iteration(self, X: np.ndarray, y: np.ndarray, estimator: BaseRegressor, iteration: int):
        super().log_iteration(X, y, estimator, iteration)
        self.pareto_fronts_.append(estimator.solution_composition_.pareto_front())
        # Very hacky way to check if were in the first Stage of TS or not
        if len(self.pareto_fronts_[-1]) > 1:
            self.log_metric("hypervolume", hypervolume(self.pareto_fronts_[-1]), estimator.step_)
            self.log_metric("spread", spread(self.pareto_fronts_[-1]), estimator.step_)
