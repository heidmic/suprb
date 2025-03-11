import numpy as np

from . import DefaultLogger
from ..base import BaseRegressor


class MOLogger(DefaultLogger):

    pareto_front_: np.ndarray

    def log_final(self, X: np.ndarray, y: np.ndarray, estimator: BaseRegressor):
        pareto_front = estimator.solution_composition_.pareto_front()
        pareto_front = np.array([solution.fitness_ for solution in pareto_front])
        self.pareto_front_ = pareto_front
        return
