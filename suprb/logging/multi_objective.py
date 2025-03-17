import numpy as np

from . import DefaultLogger
from ..base import BaseRegressor
from suprb.solution import Solution

from .metrics import hypervolume, spread


class MOLogger(DefaultLogger):

    pareto_front_: list[Solution]

    def log_final(self, X: np.ndarray, y: np.ndarray, estimator: BaseRegressor):
        super().log_final(X, y, estimator)
        self.pareto_front_ = estimator.solution_composition_.pareto_front()
        self.metrics_["hypervolume"] = hypervolume(self.pareto_front_)
        self.metrics_["spread"] = spread(self.pareto_front_)
        return
