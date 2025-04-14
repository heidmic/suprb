import numpy as np

from . import DefaultLogger
from ..base import BaseRegressor
from suprb.solution import Solution

from .metrics import hypervolume, spread


class MOLogger(DefaultLogger):

    pareto_front_: list[list[Solution]] = []

    def log_iteration(self, X: np.ndarray, y: np.ndarray, estimator: BaseRegressor, iteration: int):
        super().log_iteration(X, y, estimator, iteration)
        self.pareto_front_.append(estimator.solution_composition_.pareto_front())
        self.log_metric("hypervolume", hypervolume(self.pareto_front_), estimator.step_)
        self.log_metric("spread", spread(self.pareto_front_), estimator.step_)
