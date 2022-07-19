from abc import ABCMeta

from typing import Union

import numpy as np
from scipy.stats import halfnorm

from suprb.base import BaseComponent
from suprb.rule import Rule
from suprb.utils import RandomState
from suprb.rule.matching import MatchingFunction, OrderedBound


class RuleMutation(BaseComponent, metaclass=ABCMeta):
    """Mutates the bounds of a rule with the strength defined by sigma."""

    def __init__(self,
                 matching_type: MatchingFunction = OrderedBound,
                 sigma: Union[float, np.ndarray] = 0.1):
        self.matching_type = matching_type
        self.sigma = sigma

    def __call__(self, rule: Rule, random_state: RandomState) -> Rule:
        # Create copy of the rule
        mutated_rule = rule.clone()

        # Mutation
        self.mutate_bounds(mutated_rule, random_state)

        #### TODO move to OrderedBound components
        # Sort the bounds, because they could possibly be swapped
        mutated_rule.bounds = np.sort(mutated_rule.bounds, axis=1)

        return mutated_rule

    def mutate_bounds(self, rule: Rule, random_state: RandomState):
        pass


class SigmaRange(RuleMutation):
    """Draws the sigma used for another mutation from uniform distribution, low to high."""

    def __init__(self, mutation: RuleMutation, low: float, high: float):
        super().__init__(0)
        self.mutation = mutation
        self.low = low
        self.high = high

    def __call__(self, rule: Rule, random_state: RandomState) -> Rule:
        self.sigma = random_state.uniform(self.low, self.high) \
            if self.sigma is float \
            else random_state.uniform(self.low, self.high, len(self.sigma))
        self.mutation.sigma = self.sigma
        return self.mutation(rule, random_state)


class Normal(RuleMutation):
    """Normal noise on both bounds."""
    def __init__(self,
                 matching_type: MatchingFunction,
                 sigma: Union[float, np.ndarray]):
        super().__init__(matching_type, sigma)
        if self.matching_type is OrderedBound:
            self.mutate_bounds = self.ordered_bound

    def ordered_bound(self, rule: Rule, random_state: RandomState):
        # code inspection gives you a warning here but it is ineffectual
        rule.match.bounds += random_state.normal(scale=self.sigma,
                                        size=rule.match.bounds.shape)


class Halfnorm(RuleMutation):
    """Sample with (half)normal distribution around the center."""
    def __init__(self,
                 matching_type: MatchingFunction,
                 sigma: Union[float, np.ndarray]):
        super().__init__(matching_type, sigma)
        if self.matching_type is OrderedBound:
            self.mutate_bounds = self.ordered_bound

    def mutation(self, dimensions: int, random_state: RandomState):
        return halfnorm.rvs(scale=self.sigma / 2, size=dimensions,
                            random_state=random_state)

    def ordered_bound(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        mean = np.mean(bounds, axis=1)
        bounds[:, 0] = mean - self.mutation(dimensions=bounds.shape[0], random_state=random_state)
        bounds[:, 1] = mean + self.mutation(dimensions=bounds.shape[0], random_state=random_state)


class HalfnormIncrease(RuleMutation):
    """Increase bounds with (half)normal noise."""
    def __init__(self,
                 matching_type: MatchingFunction,
                 sigma: Union[float, np.ndarray]):
        super().__init__(matching_type, sigma)
        if self.matching_type is OrderedBound:
            self.mutate_bounds = self.ordered_bound

    def mutation(self, dimensions: int, random_state: RandomState):
        return halfnorm.rvs(scale=self.sigma / 2, size=dimensions,
                            random_state=random_state)

    def ordered_bound(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        bounds[:, 0] -= self.mutation(dimensions=bounds.shape[0], random_state=random_state)
        bounds[:, 1] += self.mutation(dimensions=bounds.shape[0], random_state=random_state)


class Uniform(RuleMutation):
    """Uniform noise on both bounds."""
    def __init__(self,
                 matching_type: MatchingFunction,
                 sigma: Union[float, np.ndarray]):
        super().__init__(matching_type, sigma)
        if self.matching_type is OrderedBound:
            self.mutate_bounds = self.ordered_bound

    def ordered_bound(self, rule: Rule, random_state: RandomState):
        rule.match.bounds += random_state.uniform(-self.sigma, self.sigma,
                                            size=rule.match.bounds.shape)


class UniformIncrease(RuleMutation):
    """Increase bounds with uniform noise."""
    def __init__(self,
                 matching_type: MatchingFunction,
                 sigma: Union[float, np.ndarray]):
        super().__init__(matching_type, sigma)
        if self.matching_type is OrderedBound:
            self.mutate_bounds = self.ordered_bound

    def mutation(self, dimensions: int, random_state: RandomState):
        return random_state.uniform(0, self.sigma, size=dimensions)

    def mutate_bounds(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        bounds[:, 0] -= self.mutation(dimensions=bounds.shape[0], random_state=random_state)
        bounds[:, 1] += self.mutation(dimensions=bounds.shape[0], random_state=random_state)
