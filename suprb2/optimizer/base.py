from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Union

import numpy as np
from sklearn.base import BaseEstimator

from suprb2.base import Solution


class BaseOptimizer(BaseEstimator, metaclass=ABCMeta):
    """Finds an optimal `Solution`."""

    def __init__(self, random_state: int, n_jobs: int):
        self.random_state = random_state
        self.n_jobs = n_jobs

    random_state_: Union[np.random.Generator, np.random.RandomState]

    @abstractmethod
    def optimize(self, X: np.ndarray, y: np.ndarray) -> Union[Solution, list[Solution], None]:
        """ Optimizes the fitness of `Individuals`.

        Parameters
        ----------
        X: np.ndarray
            Input values.
        y: np.ndarray
            Target values.

        Returns
        -------
        elitist
            Returns the best individual(s) found.
        """
        pass

    @abstractmethod
    def elitist(self):
        """ Returns the best individual.

        Returns
        -------
        elitist : Individual
            Returns the best individual found.
        """

        pass

    def _reset(self):
        """Reset internal data-dependent state of the optimizer, if necessary.

        __init__ parameters are not touched.
        """
        if hasattr(self, 'random_state_'):
            del self.random_state_

    def __str__(self):
        class_name = self.__class__
        module = class_name.__module__
        if module == 'builtins':
            return class_name.__qualname__
        return "class:" + module + '.' + class_name.__qualname__
