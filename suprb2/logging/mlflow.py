import numpy as np

from suprb2.base import BaseRegressor
from . import BaseLogger


class MlflowLogger(BaseLogger):
    """Logs relevant parameters and statistics to mlflow."""

    # TODO: Decide what to actually log to mlflow.

    def log_init(self, X: np.ndarray, y: np.ndarray, estimator: BaseRegressor):
        pass

    def log_iteration(self, X: np.ndarray, y: np.ndarray, estimator: BaseRegressor, iteration: int):
        pass

    def log_final(self, X: np.ndarray, y: np.ndarray, estimator: BaseRegressor):
        pass
