from abc import ABCMeta, abstractmethod
from typing import Union
import numpy as np
from scipy.stats import halfnorm
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from suprb.base import BaseComponent
from suprb.rule.matching import (
    MatchingFunction,
    OrderedBound,
    UnorderedBound,
    CenterSpread,
    MinPercentage,
)
from suprb.utils import check_random_state, RandomState
from . import Rule, RuleFitness
from .fitness import VolumeWu


class RuleInit(BaseComponent, metaclass=ABCMeta):
    """Generates initial `Rule`s.

    Parameters
        ----------
        bounds: np.ndarray
            The absolute bounds of the feature space.
        model: RegressorMixin
            Local model used for fitting the intervals.
    """

    def __init__(
        self,
        bounds: np.ndarray = None,
        model: RegressorMixin = None,
        fitness: RuleFitness = None,
        matching_type: MatchingFunction = None,
    ):
        self.bounds = bounds
        self.model = model
        self.fitness = fitness
        self.matching_type = matching_type

        self._validate_components(model=Ridge(alpha=0.01), fitness=VolumeWu())

    @property
    def matching_type(self):
        return self._matching_type

    @matching_type.setter
    def matching_type(self, matching_type):
        self._matching_type = matching_type
        if isinstance(self.matching_type, OrderedBound):
            self.generate_matching_function = self.ordered_bound
        elif isinstance(self.matching_type, UnorderedBound):
            self.generate_matching_function = self.unordered_bound
        elif isinstance(self.matching_type, CenterSpread):
            self.generate_matching_function = self.centre_spread
        elif isinstance(self.matching_type, MinPercentage):
            self.generate_matching_function = self.min_percentage

    def __call__(self, random_state: RandomState, mean: np.ndarray = None) -> Rule:
        """Generate a random rule.

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
        matching_function = self.generate_matching_function(mean, random_state_)
        return Rule(
            match=matching_function,
            input_space=self.bounds,
            model=clone(self.model),
            fitness=self.fitness,
        )

    def generate_matching_function(self, mean: np.ndarray, random_state: RandomState) -> MatchingFunction:
        pass

    @abstractmethod
    def ordered_bound(self, mean: np.ndarray, random_state: RandomState) -> MatchingFunction:
        pass

    @abstractmethod
    def unordered_bound(self, mean: np.ndarray, random_state: RandomState) -> MatchingFunction:
        pass

    @abstractmethod
    def centre_spread(self, mean: np.ndarray, random_state: RandomState) -> MatchingFunction:
        pass

    @abstractmethod
    def min_percentage(self, mean: np.ndarray, random_state: RandomState) -> MatchingFunction:
        pass


class MeanInit(RuleInit):
    """Initializes both bounds with the mean."""

    def ordered_bound(self, mean: np.ndarray, random_state: RandomState) -> MatchingFunction:
        return OrderedBound(np.stack((mean.T, mean.T), axis=1))

    def unordered_bound(self, mean: np.ndarray, random_state: RandomState) -> MatchingFunction:
        return UnorderedBound(np.stack((mean.T, mean.T), axis=1))

    def centre_spread(self, mean: np.ndarray, random_state: RandomState) -> MatchingFunction:
        return CenterSpread(np.stack((mean.T, np.zeros(mean.shape[0]).T), axis=1))

    def min_percentage(self, mean: np.ndarray, random_state: RandomState) -> MatchingFunction:
        return MinPercentage(np.stack((mean.T, np.zeros(mean.shape[0]).T), axis=1))


class NormalInit(RuleInit):
    """Initializes both bounds with points drawn from a normal distribution."""

    def __init__(
        self,
        bounds: np.ndarray = None,
        model: RegressorMixin = None,
        fitness: RuleFitness = None,
        matching_type: MatchingFunction = None,
        sigma: Union[float, np.ndarray] = 0.1,
    ):
        super().__init__(bounds=bounds, model=model, fitness=fitness, matching_type=matching_type)
        self.sigma = sigma
        if self.matching_type in (CenterSpread, MinPercentage):
            assert isinstance(self.sigma, np.ndarray) and self.sigma.shape[0] == 2

    def sample_individual_bounds(self, mean: np.ndarray, random_state: RandomState):
        allele_1 = random_state.normal(loc=mean, scale=self.sigma[0], size=(mean.shape[0]))
        allele_2 = halfnorm.rvs(scale=self.sigma[1] / 2, size=mean.shape[0], random_state=random_state)
        return np.stack((allele_1, allele_2), axis=1)

    def ordered_bound(self, mean: np.ndarray, random_state: RandomState) -> MatchingFunction:
        return OrderedBound(
            np.sort(
                random_state.normal(loc=mean, scale=self.sigma, size=(2, mean.shape[0])).T,
                axis=1,
            )
        )

    def unordered_bound(self, mean: np.ndarray, random_state: RandomState) -> MatchingFunction:
        return UnorderedBound(random_state.normal(loc=mean, scale=self.sigma, size=(2, mean.shape[0])).T)

    def centre_spread(self, mean: np.ndarray, random_state: RandomState) -> MatchingFunction:
        return CenterSpread(self.sample_individual_bounds(mean, random_state))

    def min_percentage(self, mean: np.ndarray, random_state: RandomState) -> MatchingFunction:
        return MinPercentage(self.sample_individual_bounds(mean, random_state))


class HalfnormInit(RuleInit):
    """Initializes both bounds with points drawn from a halfnorm distribution, so that the mean is always matched."""

    def __init__(
        self,
        bounds: np.ndarray = None,
        model: RegressorMixin = None,
        fitness: RuleFitness = None,
        matching_type: MatchingFunction = None,
        sigma: float = 0.1,
    ):
        super().__init__(bounds=bounds, model=model, fitness=fitness, matching_type=matching_type)
        self.sigma = sigma

    def sample_bounds(self, mean: np.ndarray, random_state: RandomState):
        low = mean - halfnorm.rvs(scale=self.sigma, size=mean.shape[0], random_state=random_state)
        high = mean + halfnorm.rvs(scale=self.sigma, size=mean.shape[0], random_state=random_state)
        return np.stack((low.T, high.T), axis=1)

    def ordered_bound(self, mean: np.ndarray, random_state: RandomState) -> MatchingFunction:
        return OrderedBound(self.sample_bounds(mean, random_state))

    def unordered_bound(self, mean: np.ndarray, random_state: RandomState) -> MatchingFunction:
        return UnorderedBound(self.sample_bounds(mean, random_state))

    def centre_spread(self, mean: np.ndarray, random_state: RandomState) -> MatchingFunction:
        raise TypeError("Halform Init is not implemented for CSR")

    def min_percentage(self, mean: np.ndarray, random_state: RandomState) -> MatchingFunction:
        raise TypeError("Halform Init is not implemented for MPR")
