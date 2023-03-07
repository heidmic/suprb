import warnings

import numpy as np
from sklearn import clone

from suprb import SupRB
from suprb.base import BaseRegressor
from . import BaseLogger


class CombinedLogger(BaseLogger):
    """This logger defines an interface for combining multiple loggers."""

    loggers_: list[tuple[str, BaseLogger]]

    def __init__(self, loggers: list[tuple[str, BaseLogger]]):
        """An unique name for every logger must be supplied, such that the parameter get/set are well-defined."""
        self.loggers = loggers

    def log_init(self, X: np.ndarray, y: np.ndarray, estimator: BaseRegressor):
        if any(map(lambda logger: isinstance(logger, CombinedLogger), self.loggers)):
            warnings.warn("Nesting loggers is not recommended. Please add all loggers to this top-level logger.")

        self.loggers_ = [(name, clone(logger)) for name, logger in self.loggers]

        for _, logger in self.loggers_:
            logger.log_init(X=X, y=y, estimator=estimator)

    def log_iteration(self, X: np.ndarray, y: np.ndarray, estimator: BaseRegressor, iteration: int):
        for _, logger in self.loggers_:
            logger.log_iteration(X=X, y=y, estimator=estimator, iteration=iteration)

    def log_final(self, X: np.ndarray, y: np.ndarray, estimator: BaseRegressor):
        for _, logger in self.loggers_:
            logger.log_final(X=X, y=y, estimator=estimator)

    def get_params(self, deep=True):
        # Call get_params on every logger
        out = super().get_params(deep=deep)
        if not deep:
            return out
        out.update(self.loggers)
        for name, logger in self.loggers:
            if hasattr(logger, "get_params"):
                for key, value in logger.get_params(deep=True).items():
                    out["%s__%s" % (name, key)] = value
        return out

    def set_params(self, **params):
        # Ensure strict ordering of parameter setting:
        # 1. All steps
        if 'loggers' in params:
            self.loggers = params.pop('loggers')
        # 2. Step replacement
        items = self.loggers
        names = []
        if items:
            names, _ = zip(*items)
        for name in list(params.keys()):
            if "__" not in name and name in names:
                self._replace_logger(name, params.pop(name))
        # 3. Step parameters and other initialisation arguments
        super().set_params(**params)
        return self

    def _replace_logger(self, name: str, new_val: BaseLogger):
        # assumes `name` is a valid logger name
        new_loggers = list(self.loggers)
        for i, (logger_name, _) in enumerate(new_loggers):
            if logger_name == name:
                new_loggers[i] = (name, new_val)
                break
        self.loggers = new_loggers

    def get_elitist(self, estimator: SupRB):
        pass
