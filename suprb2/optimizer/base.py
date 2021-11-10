from abc import ABCMeta, abstractmethod
from typing import Union

import numpy as np
from sklearn.base import BaseEstimator


class Solution(metaclass=ABCMeta):
    """An individual solution to some problem."""

    is_fitted_: bool
    error_: float
    fitness_: float

    def __str__(self):
        if hasattr(self, 'is_fitted_') and self.is_fitted_:
            attributes = {'error': self.error_, 'fitness': self.fitness_} | self._more_str_attributes()
            concat = ",".join([f"{key}={value}" for key, value in attributes.items()])
            return f"{self.__class__.__name__}({concat})"

    def _more_str_attributes(self) -> dict:
        return {}


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
