from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Union

import numpy as np
from sklearn.base import RegressorMixin, clone
from sklearn.metrics import mean_squared_error

from suprb2.base import Solution
from suprb2.fitness import BaseFitness


class RuleFitness(BaseFitness, metaclass=ABCMeta):
    """Evaluates the fitness of a `Rule`."""

    @abstractmethod
    def __call__(self, rule: Rule) -> float:
        pass


class Rule(Solution):
    """ A rule that fits the input data in a certain interval.

    Parameters
    ----------
    bounds: np.ndarray
        The interval this rule will be fitted on.
    input_space: np.ndarray
        The bounds of the input space `X`.
    model: RegressorMixin
        Local model used to fit an interval.
    """

    experience_: float
    match_: np.ndarray
    pred_: Union[np.ndarray, None]  # only the prediction of matching points, so of x[match_]

    def __init__(self, bounds: np.ndarray, input_space: np.ndarray, model: RegressorMixin, fitness: RuleFitness):
        self.bounds = bounds
        self.input_space = input_space
        self.model = model
        self.fitness = fitness

    def fit(self, X: np.ndarray, y: np.ndarray) -> Rule:

        # Match input data
        match = self.matched_data(X)

        # No reason to fit if no data point matches
        if not np.any(match):
            self.is_fitted_ = False
            self.error_ = np.inf
            self.fitness_ = -np.inf
            self.experience_ = 0
            self.pred_ = np.array([])
            self.match_ = match
            return self

        # No reason to refit if matched data points did not change
        if hasattr(self, 'match_'):
            if (self.match_ == match).all():
                self.is_fitted_ = True
                return self

        self.match_ = match

        # Get all data points which match the bounds.
        X, y = X[self.match_], y[self.match_]

        # Create and fit the model
        self.model.fit(X, y)

        self.pred_ = self.model.predict(X)
        self.error_ = max(mean_squared_error(y, self.pred_), 1e-4)  # TODO: make min a parameter?
        self.fitness_ = self.fitness(self)
        self.experience_ = float(X.shape[0])

        self.is_fitted_ = True
        return self

    def matched_data(self, X: np.ndarray):
        """Returns a boolean array that is True for data points the rule matches."""
        return np.all((self.bounds[:, 0] <= X) & (X <= self.bounds[:, 1]), axis=1)

    @property
    def volume_(self):
        """Calculates the volume of the interval."""
        diff = self.bounds[:, 1] - self.bounds[:, 0]
        return np.prod(diff)

    def predict(self, X: np.ndarray):
        return self.model.predict(X)

    def clone(self, **kwargs) -> Rule:
        args = dict(
            bounds=self.bounds.copy() if 'bounds' not in kwargs else None,
            input_space=self.input_space,
            model=clone(self.model) if 'model' not in kwargs else None,
            fitness=self.fitness
        )
        return Rule(**(args | kwargs))

    def _validate_bounds(self, X: np.ndarray):
        """Validates that bounds have the correct shape."""

        if self.bounds.shape[1] != 2:
            raise ValueError(f"specified bounds are not of shape (-1, 2), but {self.bounds.shape}")

        if self.bounds.shape[0] != X.shape[1]:
            raise ValueError(f"bounds- and input data dimension mismatch: {self.bounds.shape[0]} != {X.shape[1]}")

    def _more_str_attributes(self) -> dict:
        return {'experience': self.experience_}
