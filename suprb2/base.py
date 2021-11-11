from __future__ import annotations

from abc import abstractmethod, ABCMeta

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class BaseComponent(BaseEstimator, metaclass=ABCMeta):
    """
    Base class for subcomponents of a model.
    Note that this technically goes against scikit-learn standards, because input parameters should be
    non-modifiable, but as many subcomponents have different parameters, it is just not possible
    to write all of them into the __init__() of the top-level model.
    This hierarchy of nested components is much cleaner for this use-case.
    The necessary interface to other scikit-learn functionality is still present, because all components
    are instantiated internally, so `sklearn.utils.estimator_checks.check_estimator()` should still pass.
    """
    pass


class BaseRegressor(BaseEstimator, RegressorMixin, metaclass=ABCMeta):
    """A base (composite) Regressor."""

    is_fitted_: bool

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """ A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values.

        Returns
        -------
        self : BaseEstimator
            Returns self.
        """

        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : np.ndarray
            Returns the estimation with shape (n_samples,).
        """

        pass
