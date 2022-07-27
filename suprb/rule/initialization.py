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

    def __call__(self, random_state: RandomState, mean: np.ndarray = None) -> Rule:
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

        return Rule(bounds=bounds, input_space=self.bounds, model=clone(self.model), fitness=self.fitness)

    @abstractmethod
    def generate_bounds(self, mean: np.ndarray, random_state: RandomState) -> np.ndarray:
        pass


class MeanInit(RuleInit):
    """Initializes the center with the mean and sets the deviations to 0."""

    def generate_bounds(self, mean: np.ndarray, _random_state: RandomState) -> np.ndarray:
        return np.stack((mean.T, np.zeros((mean.shape[0]))), axis=1)


class NormalInit(RuleInit):
    """Initializes center with points drawn from a normal distribution and deviations with
    a halfnorm distribution scaled with sigma_deviations ."""

    def __init__(self, bounds: np.ndarray = None, model: RegressorMixin = None, fitness: RuleFitness = None,
                 sigma_center: float = 0.1, sigma_deviations: float = 0.01):
        super().__init__(bounds=bounds, model=model, fitness=fitness)
        self.sigma_center = sigma_center
        self.sigma_deviations = sigma_deviations

    def generate_bounds(self, mean: np.ndarray, random_state: RandomState) -> np.ndarray:
        center = random_state.normal(loc=mean, scale=self.sigma_center, size=(mean.shape[0]))
        deviations = halfnorm.rvs(scale=self.sigma_deviations / 2, size=mean.shape[0], random_state=random_state)
        return np.stack((center.T, deviations.T), axis=1)