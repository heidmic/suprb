from abc import abstractmethod

import numpy as np

from suprb2.base import BaseComponent, BaseRegressor


class BaseLogger(BaseComponent):
    """The base class for loggers."""

    @abstractmethod
    def log_init(self, X: np.ndarray, y: np.ndarray, estimator: BaseRegressor):
        """Logs initial parameters, before any iteration or fitting has taken place."""
        pass

    @abstractmethod
    def log_iteration(self, X: np.ndarray, y: np.ndarray, estimator: BaseRegressor, iteration: int):
        """Logs an iteration of the estimator. May actually compute errors or scores, depending on the state."""
        pass

    @abstractmethod
    def log_final(self, X: np.ndarray, y: np.ndarray, estimator: BaseRegressor):
        """Log the final state of the estimator. It is assumed that the fitting process is already completed."""
        pass

    def __str__(self):
        class_name = self.__class__
        module = class_name.__module__
        if module == 'builtins':
            return class_name.__qualname__
        return "class:" + module + '.' + class_name.__qualname__
