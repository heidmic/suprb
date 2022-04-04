from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Union

import numpy as np
from sklearn.base import BaseEstimator

from suprb.base import SolutionBase
from suprb.utils import RandomState


class BaseOptimizer(BaseEstimator, metaclass=ABCMeta):
    """Finds an optimal `Solution`."""

    def __init__(self, random_state: int, n_jobs: int):
        self.random_state = random_state
        self.n_jobs = n_jobs

    random_state_: RandomState

    @abstractmethod
    def optimize(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Union[SolutionBase, list[SolutionBase], None]:
        """ Optimizes the fitness of `Solutions`.

        Parameters
        ----------
        X: np.ndarray
            Input values.
        y: np.ndarray
            Target values.

        Returns
        -------
        elitist
            Returns the best solution(s) found.
        """
        pass

    def _reset(self):
        """Reset internal data-dependent state of the optimizer, if necessary.

        __init__ parameters are not touched.
        """
        if hasattr(self, 'random_state_'):
            del self.random_state_
