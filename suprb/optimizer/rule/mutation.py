from abc import ABCMeta, abstractmethod

from typing import Union

import numpy as np
from scipy.stats import halfnorm

from suprb.base import BaseComponent
from suprb.rule import Rule
from suprb.utils import RandomState
from suprb.rule.matching import MatchingFunction, OrderedBound, UnorderedBound, CentreSpread, MinPercentage


class RuleMutation(BaseComponent, metaclass=ABCMeta):
    """Mutates the bounds of a rule with the strength defined by sigma."""

    def __init__(self,
                 matching_type: MatchingFunction = None,
                 sigma: Union[float, np.ndarray] = 0.1,
                 sigma_two: Union[float, np.ndarray] = 0.1):
        self.matching_type = matching_type
        self.sigma = sigma
        self.sigma_two = sigma_two

    @property
    def matching_type(self):
        return self._matching_type

    @matching_type.setter
    def matching_type(self, matching_type):
        self._matching_type = matching_type
        if isinstance(self.matching_type, OrderedBound):
            self.mutate_bounds = self.ordered_bound
        elif isinstance(self.matching_type, UnorderedBound):
            self.mutate_bounds = self.unordered_bound
        elif isinstance(self.matching_type, CentreSpread):
            self.mutate_bounds = self.centre_spread
        elif isinstance(self.matching_type, MinPercentage):
            self.mutate_bounds = self.min_percentage

    def __call__(self, rule: Rule, random_state: RandomState) -> Rule:
        # Create copy of the rule
        mutated_rule = rule.clone()

        # Mutation
        self.mutate_bounds(mutated_rule, random_state)

        return mutated_rule

    def mutate_bounds(self, rule: Rule, random_state: RandomState):
        pass

    @abstractmethod
    def ordered_bound(self, rule: Rule, random_state: RandomState):
        pass

    @abstractmethod
    def unordered_bound(self, rule: Rule, random_state: RandomState):
        pass

    @abstractmethod
    def centre_spread(self, rule: Rule, random_state: RandomState):
        pass

    @abstractmethod
    def min_percentage(self, rule: Rule, random_state: RandomState):
        pass


class SigmaRange(RuleMutation):
    """Draws the sigma used for another mutation from uniform distribution, low to high."""

    def __init__(self, mutation: RuleMutation, low: float, high: float):
        super().__init__(0)
        self.mutation = mutation
        self.low = low
        self.high = high

    def __call__(self, rule: Rule, random_state: RandomState) -> Rule:
        uniform_size = None if isinstance(self.sigma, float) else len(self.sigma)
        self.sigma = random_state.uniform(self.low, self.high, uniform_size)
        self.mutation.sigma = self.sigma
        return self.mutation(rule, random_state)


class Normal(RuleMutation):
    """Normal noise on both bounds."""

    def ordered_bound(self, rule: Rule, random_state: RandomState):
        # code inspection gives you a warning here but it is ineffectual
        rule.match.bounds += random_state.normal(scale=self.sigma,
                                                 size=rule.match.bounds.shape)
        rule.match.bounds = np.sort(rule.match.bounds, axis=1)

    def unordered_bound(self, rule: Rule, random_state: RandomState):
        rule.match.bounds += random_state.normal(scale=self.sigma,
                                                 size=rule.match.bounds.shape)

    def centre_spread(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        bounds[:, 0] += random_state.normal(scale=self.sigma,
                                            size=rule.match.bounds.shape[0])
        bounds[:, 1] += random_state.normal(scale=self.sigma_two,
                                            size=rule.match.bounds.shape[0])

    def min_percentage(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        bounds[:, 0] += random_state.normal(scale=self.sigma,
                                            size=rule.match.bounds.shape[0])
        bounds[:, 1] += random_state.normal(scale=self.sigma_two,
                                            size=rule.match.bounds.shape[0])


class Halfnorm(RuleMutation):
    """Sample with (half)normal distribution around the center."""

    def ordered_bound(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        mean = np.mean(bounds, axis=1)
        bounds[:, 0] = mean - \
            halfnorm.rvs(scale=self.sigma / 2, size=rule.bounds.shape[0], random_state=random_state)

        bounds[:, 1] = mean + \
            halfnorm.rvs(scale=self.sigma / 2, size=rule.bounds.shape[0], random_state=random_state)
        rule.match.bounds = np.sort(rule.match.bounds, axis=1)

    def unordered_bound(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        mean = np.mean(bounds, axis=1)
        bounds[:, 0] = mean - \
            halfnorm.rvs(scale=self.sigma / 2, size=rule.bounds.shape[0], random_state=random_state)

        bounds[:, 1] = mean + \
            halfnorm.rvs(scale=self.sigma / 2, size=rule.bounds.shape[0], random_state=random_state)

    def centre_spread(self, rule: Rule, random_state: RandomState):
        pass

    def min_percentage(self, rule: Rule, random_state: RandomState):
        pass


class HalfnormIncrease(RuleMutation):
    """Increase bounds with (half)normal noise."""

    def ordered_bound(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        bounds[:, 0] -= halfnorm.rvs(scale=self.sigma / 2, size=bounds.shape[0], random_state=random_state)
        bounds[:, 1] += halfnorm.rvs(scale=self.sigma / 2, size=bounds.shape[0], random_state=random_state)
        rule.match.bounds = np.sort(rule.match.bounds, axis=1)

    def unordered_bound(self, rule: Rule, random_state: RandomState):
        pass

    def centre_spread(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        bounds[:, 0] += random_state.normal(scale=self.sigma, size=bounds.shape[0])
        bounds[:, 1] += halfnorm.rvs(scale=self.sigma_two / 2, size=bounds.shape[0], random_state=random_state)

    def min_percentage(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        bounds[:, 0] -= halfnorm.rvs(scale=self.sigma / 2, size=bounds.shape[0], random_state=random_state)
        bounds[:, 1] += halfnorm.rvs(scale=self.sigma_two / 2, size=bounds.shape[0], random_state=random_state)


class Uniform(RuleMutation):
    """Uniform noise on both bounds."""

    def ordered_bound(self, rule: Rule, random_state: RandomState):
        rule.match.bounds += random_state.uniform(-self.sigma, self.sigma,
                                                  size=rule.match.bounds.shape)
        rule.match.bounds = np.sort(rule.match.bounds, axis=1)

    def unordered_bound(self, rule: Rule, random_state: RandomState):
        rule.match.bounds += random_state.uniform(-self.sigma, self.sigma,
                                                  size=rule.match.bounds.shape)

    def centre_spread(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        bounds[:, 0] += random_state.uniform(-self.sigma, self.sigma, size=bounds.shape[0])
        bounds[:, 1] += random_state.uniform(-self.sigma_two, self.sigma_two, size=bounds.shape[0])

    def min_percentage(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        bounds[:, 0] += random_state.uniform(-self.sigma, self.sigma, size=bounds.shape[0])
        bounds[:, 1] += random_state.uniform(-self.sigma_two, self.sigma_two, size=bounds.shape[0])


class UniformIncrease(RuleMutation):
    """Increase bounds with uniform noise."""

    def mutation(self, dimensions: int, random_state: RandomState):
        return random_state.uniform(0, self.sigma, size=dimensions)

    def ordered_bound(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        bounds[:, 0] -= random_state.uniform(0, self.sigma, size=bounds.shape[0])
        bounds[:, 1] += random_state.uniform(0, self.sigma, size=bounds.shape[0])
        rule.match.bounds = np.sort(rule.match.bounds, axis=1)

    def unordered_bound(self, rule: Rule, random_state: RandomState):
        pass

    def centre_spread(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        bounds[:, 0] -= random_state.uniform(-self.sigma, self.sigma, size=bounds.shape[0])
        bounds[:, 1] += random_state.uniform(0, self.sigma_two, size=bounds.shape[0])
        rule.match.bounds = np.sort(rule.match.bounds, axis=1)

    def min_percentage(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        bounds[:, 0] -= random_state.uniform(0, self.sigma, size=bounds.shape[0])
        bounds[:, 1] += random_state.uniform(0, self.sigma_two, size=bounds.shape[0])
        rule.match.bounds = np.sort(rule.match.bounds, axis=1)
