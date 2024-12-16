from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Union

import numpy as np
from sklearn.base import RegressorMixin, ClassifierMixin, clone
from sklearn.metrics import mean_squared_error, accuracy_score

from suprb.base import SolutionBase
from suprb.fitness import BaseFitness
from .matching import MatchingFunction


class RuleFitness(BaseFitness, metaclass=ABCMeta):
    """Evaluates the fitness of a `Rule`."""

    @abstractmethod
    def __call__(self, rule: Rule) -> float:
        pass


class Rule(SolutionBase):
    """ A rule that fits the input data in a certain interval.

    Parameters
    ----------
    match: MatchingFunction
        The function this rule will use the determine the subset of input
        data that this rule governs (will be fitted on and predict)
    input_space: np.ndarray
        The bounds of the input space `X`.
    model: RegressorMixin
        Local model used to fit an interval.
    """

    experience_: float
    match_set_: np.ndarray
    pred_: Union[np.ndarray, None]  # only the prediction of matching points, so of x[match_]

    def __init__(self, match: MatchingFunction, input_space: np.ndarray, model: RegressorMixin | ClassifierMixin, fitness: RuleFitness):
        self.match = match
        self.input_space = input_space
        self.model = model
        self.fitness = fitness
        if isinstance(model, ClassifierMixin):
            self.task = 'Classification'
        else:
            self.task = 'Regression'

    def fit(self, X: np.ndarray, y: np.ndarray) -> Rule:

        # Match input data
        match_set = self.match(X)

        # No reason to fit if no data point matches
        if not np.any(match_set):
            self.is_fitted_ = False
            self.score_ = np.inf
            self.fitness_ = -np.inf
            self.experience_ = 0
            self.pred_ = np.array([])
            self.match_set_ = match_set
            return self

        # No reason to refit if matched data points did not change
        if hasattr(self, 'match_set_'):
            if (self.match_set_ == match_set).all():
                self.is_fitted_ = True
                return self

        self.match_set_ = match_set

        # Get all data points which match the bounds.
        X, y = X[self.match_set_], y[self.match_set_]

        # No reason to fit if only one class match_set
        if self.task == 'Classification':
            if len(np.unique(y)) == 1:
                self.pred_ = y[0]
                self.score_ = 1 - 1e-4
                self.fitness_ = self.fitness(self)
                self.experience_ = float(X.shape[0])
                self.is_fitted_ = True
                return self

        # Create and fit the model
        self.model.fit(X, y)

        self.pred_ = self.model.predict(X)
        if self.task == 'Regression':
            self.score_ = max(mean_squared_error(y, self.pred_), 1e-4)  # TODO: make min a parameter?
        elif self.task == 'Classification':
            self.score_ = -max(accuracy_score(y, self.pred_), 1e-4)
        self.fitness_ = self.fitness(self)
        self.experience_ = float(X.shape[0])

        self.is_fitted_ = True
        return self

    @property
    def volume_(self):
        return self.match.volume_

    def predict(self, X: np.ndarray):
        return self.model.predict(X)

    def clone(self, **kwargs) -> Rule:
        args = dict(
            match=self.match.copy() if 'match' not in kwargs else None,
            input_space=self.input_space,
            model=clone(self.model) if 'model' not in kwargs else None,
            fitness=self.fitness
        )
        return Rule(**(args | kwargs))

    def _more_str_attributes(self) -> dict:
        return {'experience': self.experience_}
