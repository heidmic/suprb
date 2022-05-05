from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import halfnorm
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from suprb.base import BaseComponent
from suprb.utils import check_random_state, RandomState
from . import Rule, RuleFitness
from .fitness import VolumeWu


class RuleInit(BaseComponent, metaclass=ABCMeta):
    """ Generates initial `Rule`s.

    Parameters
        ----------
        bounds: np.ndarray
            The generated interval will lie within the absolute bounds.
        model: RegressorMixin
            Local model used for fitting the intervals.
    """

    def __init__(self, bounds: np.ndarray = None, model: RegressorMixin = None, fitness: RuleFitness = None):
        self.bounds = bounds
        self.model = model
        self.fitness = fitness

        self._validate_components(model=Ridge(alpha=0.01), fitness=VolumeWu())

    def __call__(self, random_state: RandomState, mean: np.ndarray = None, dummy_rules: bool = False) -> Rule:
        """ Generate a random rule.

        Parameters
        ----------
        mean: np.ndarray
            Mean of the normal distribution to draw from.
        random_state : int, RandomState instance or None, default=None
            Pass an int for reproducible results across multiple function calls.
        """

        random_state_ = check_random_state(random_state)

        # Place the center of the rules uniformly distributed
        if mean is None:
            mean = random_state_.uniform(self.bounds[:, 0], self.bounds[:, 1])
        # Sample the bounds
        bounds = self.generate_bounds(mean, random_state_)
        bounds = np.sort(bounds, axis=1)

        if dummy_rules:
            low = np.full(mean.shape[0], 0.0)
            high = np.full(mean.shape[0], 0.0)
            bounds = np.stack((low, high), axis=1)

        return Rule(bounds=bounds, input_space=self.bounds, model=clone(self.model), fitness=self.fitness)

    @abstractmethod
    def generate_bounds(self, mean: np.ndarray, random_state: RandomState) -> np.ndarray:
        pass


class MeanInit(RuleInit):
    """Initializes both bounds with the mean."""

    def generate_bounds(self, mean: np.ndarray, _random_state: RandomState) -> np.ndarray:
        return np.stack((mean.T, mean.T), axis=1)


class NormalInit(RuleInit):
    """Initializes both bounds with points drawn from a normal distribution."""

    def __init__(self, bounds: np.ndarray = None, model: RegressorMixin = None, fitness: RuleFitness = None,
                 sigma: float = 0.1):
        super().__init__(bounds=bounds, model=model, fitness=fitness)
        self.sigma = sigma

    def generate_bounds(self, mean: np.ndarray, random_state: RandomState) -> np.ndarray:
        return random_state.normal(loc=mean, scale=self.sigma, size=(2, mean.shape[0])).T


class HalfnormInit(RuleInit):
    """Initializes both bounds with points drawn from a halfnorm distribution, so that the mean is always matched."""

    def __init__(self, bounds: np.ndarray = None, model: RegressorMixin = None, fitness: RuleFitness = None,
                 sigma: float = 0.1):
        super().__init__(bounds=bounds, model=model, fitness=fitness)
        self.sigma = sigma

    def generate_bounds(self, mean: np.ndarray, random_state: RandomState) -> np.ndarray:
        low = mean - halfnorm.rvs(scale=self.sigma, size=mean.shape[0], random_state=random_state)
        high = mean + halfnorm.rvs(scale=self.sigma, size=mean.shape[0], random_state=random_state)
        return np.stack((low.T, high.T), axis=1)
