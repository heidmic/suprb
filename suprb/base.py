from __future__ import annotations

from abc import abstractmethod, ABCMeta, ABC

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


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

    def _validate_components(self, **kwargs):
        for parameter_name, default in kwargs.items():
            current_value = self.__getattribute__(parameter_name)
            self.__setattr__(parameter_name, current_value if current_value is not None else default)


class SolutionBase(metaclass=ABCMeta):
    """An individual solution to some problem."""

    is_fitted_: bool
    error_: float
    fitness_: float

    @abstractmethod
    def clone(self, **kwargs) -> SolutionBase:
        """Clone a solution such that all relevant attributes are copied / transferred to the new solution."""
        pass

    def __str__(self):
        if hasattr(self, 'is_fitted_') and self.is_fitted_:
            attributes = {'error': self.error_, 'fitness': self.fitness_} | self._more_str_attributes()
            concat = ",".join([f"{key}={value}" for key, value in attributes.items()])
            return f"{self.__class__.__name__}({concat})"

    def _more_str_attributes(self) -> dict:
        """Should return name and value of additional attributes that should be included in the str representation."""
        return {}


class BaseSupervised(BaseEstimator, metaclass=ABCMeta):
    """A base (composite) Estimator for supervised learning."""

    is_fitted_: bool

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseSupervised:
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

class BaseRegressor(BaseSupervised, RegressorMixin, metaclass=ABCMeta):
    """A base (composite) Regressor."""

    is_fitted_: bool

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseRegressor:
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


class BaseClassifier(BaseSupervised, ClassifierMixin, metaclass=ABCMeta):
    """A base (composite) Classifier."""

    is_fitted_: bool

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseClassifier:
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

class SupervisedMixin(ABC):
    pass

SupervisedMixin.register(RegressorMixin)
SupervisedMixin.register(ClassifierMixin)