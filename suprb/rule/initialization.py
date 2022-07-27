from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import halfnorm
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


from suprb.base import BaseComponent
from suprb.rule.matching import MatchingFunction, OrderedBound
from suprb.utils import check_random_state, RandomState
from . import Rule, RuleFitness
from .fitness import VolumeWu


class RuleInit(BaseComponent, metaclass=ABCMeta):
    """ Generates initial `Rule`s.

    Parameters
        ----------
        bounds: np.ndarray
            The absolute bounds of the feature space.
        model: RegressorMixin
            Local model used for fitting the intervals.
    """

    def __init__(self, bounds: np.ndarray = None,
                 model: RegressorMixin = None,
                 fitness: RuleFitness = None,
                 matching_type: MatchingFunction = None):
        self.bounds = bounds
        self.model = model
        self.fitness = fitness
        self.matching_type = matching_type

        self._validate_components(model=Ridge(alpha=0.01),
                                  fitness=VolumeWu())

    @property
    def matching_type(self):
        return self._matching_type

    @matching_type.setter
    def matching_type(self, matching_type):
        self._matching_type = matching_type
        if isinstance(self.matching_type, OrderedBound):
            self.generate_matchf = self.ordered_bound

    def __call__(self, random_state: RandomState, mean: np.ndarray = None) -> Rule:
        """ Generate a random rule.

        Parameters
        ----------
        mean: np.ndarray
            Mean of the normal distribution to draw from. If none was
            provided it is drawn randomly from a uniform distribution.
            Should lie within the absolute bounds of the feature space.
        random_state : int, RandomState instance or None, default=None
            Pass an int for reproducible results across multiple function calls.
        """

        random_state_ = check_random_state(random_state)

        # Place the center of the rules uniformly distributed
        if mean is None:
            mean = random_state_.uniform(self.bounds[:, 0], self.bounds[:, 1])

        # Sample the bounds
        matchf = self.generate_matchf(mean, random_state_)
        return Rule(match=matchf, input_space=self.bounds, model=clone(
            self.model), fitness=self.fitness)

    def generate_matchf(self, mean: np.ndarray, random_state: RandomState) -> \
            MatchingFunction:
        pass

    @abstractmethod
    def ordered_bound(self, mean: np.ndarray, random_state: RandomState):
        pass


class MeanInit(RuleInit):
    """Initializes both bounds with the mean."""

    def ordered_bound(self, mean: np.ndarray, random_state: RandomState) -> MatchingFunction:
        return OrderedBound(np.stack((mean.T, mean.T), axis=1))


class NormalInit(RuleInit):
    """Initializes both bounds with points drawn from a normal distribution."""

    def __init__(self, bounds: np.ndarray = None, model: RegressorMixin = None, fitness: RuleFitness = None,
                 sigma: float = 0.1):
        super().__init__(bounds=bounds, model=model, fitness=fitness)
        self.sigma = sigma

    def ordered_bound(self, mean: np.ndarray, random_state: RandomState) -> MatchingFunction:
        return OrderedBound(np.sort(random_state.normal(loc=mean,
                                                        scale=self.sigma,
                                                        size=(2, mean.shape[0])).T,
                                    axis=1))


class HalfnormInit(RuleInit):
    """Initializes both bounds with points drawn from a halfnorm distribution, so that the mean is always matched."""

    def __init__(self, bounds: np.ndarray = None, model: RegressorMixin = None, fitness: RuleFitness = None,
                 sigma: float = 0.1):
        super().__init__(bounds=bounds, model=model, fitness=fitness)
        self.sigma = sigma

    def ordered_bound(self, mean: np.ndarray, random_state: RandomState) -> MatchingFunction:
        low = mean - halfnorm.rvs(scale=self.sigma, size=mean.shape[0], random_state=random_state)
        high = mean + halfnorm.rvs(scale=self.sigma, size=mean.shape[0], random_state=random_state)
        return OrderedBound(np.stack((low.T, high.T), axis=1))
