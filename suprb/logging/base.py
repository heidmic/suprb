from abc import abstractmethod

import numpy as np

from suprb.base import BaseComponent, BaseSupervised


class BaseLogger(BaseComponent):
    """The base class for loggers."""

    @abstractmethod
    def log_init(self, X: np.ndarray, y: np.ndarray, estimator: BaseSupervised):
        """Logs initial parameters, before any iteration or fitting has taken place."""
        pass

    @abstractmethod
    def log_iteration(self, X: np.ndarray, y: np.ndarray, estimator: BaseSupervised, iteration: int):
        """Logs an iteration of the estimator. May actually compute errors or scores, depending on the state."""
        pass

    @abstractmethod
    def log_final(self, X: np.ndarray, y: np.ndarray, estimator: BaseSupervised):
        """Log the final state of the estimator. It is assumed that the fitting process is already completed."""
        pass

    @abstractmethod
    def get_elitist(self, estimator: BaseSupervised):
        """Log the final elitist"""
        pass
