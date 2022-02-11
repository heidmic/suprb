from abc import ABCMeta

import numpy as np
from scipy.stats import halfnorm

from suprb2.base import BaseComponent
from suprb2.rule import Rule
from suprb2.utils import RandomState


class RuleMutation(BaseComponent, metaclass=ABCMeta):
    """Mutates the bounds of a rule with the strength defined by sigma."""

    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def __call__(self, rule: Rule, random_state: RandomState) -> Rule:
        # Create copy of the rule
        mutated_rule = rule.clone()

        # Mutation
        self.mutate_bounds(mutated_rule, random_state)

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
        self.sigma = random_state.uniform(self.low, self.high)
        self.mutation.sigma = self.sigma
        return self.mutation(rule, random_state)


class Normal(RuleMutation):
    """Normal noise on both bounds."""

    def mutate_bounds(self, rule: Rule, random_state: RandomState):
        rule.bounds += random_state.normal(scale=self.sigma, size=rule.bounds.shape)


class Halfnorm(RuleMutation):
    """Sample with (half)normal distribution around the center."""

    def mutation(self, rule: Rule, random_state: RandomState):
        return halfnorm.rvs(scale=self.sigma / 2, size=rule.bounds.shape[0], random_state=random_state)

    def mutate_bounds(self, rule: Rule, random_state: RandomState):
        mean = np.mean(rule.bounds, axis=1)
        rule.bounds[:, 0] = mean - self.mutation(rule=rule, random_state=random_state)
        rule.bounds[:, 1] = mean + self.mutation(rule=rule, random_state=random_state)


class HalfnormIncrease(RuleMutation):
    """Increase bounds with (half)normal noise."""

    def mutation(self, rule: Rule, random_state: RandomState):
        return halfnorm.rvs(scale=self.sigma / 2, size=rule.bounds.shape[0], random_state=random_state)

    def mutate_bounds(self, rule: Rule, random_state: RandomState):
        rule.bounds[:, 0] -= self.mutation(rule=rule, random_state=random_state)
        rule.bounds[:, 1] += self.mutation(rule=rule, random_state=random_state)


class Uniform(RuleMutation):
    """Uniform noise on both bounds."""

    def mutate_bounds(self, rule: Rule, random_state: RandomState):
        rule.bounds += random_state.uniform(-self.sigma, self.sigma, size=rule.bounds.shape)


class UniformIncrease(RuleMutation):
    """Increase bounds with uniform noise."""

    def mutation(self, rule: Rule, random_state: RandomState):
        return random_state.uniform(0, self.sigma, size=rule.bounds.shape[0])

    def mutate_bounds(self, rule: Rule, random_state: RandomState):
        rule.bounds[:, 0] -= self.mutation(rule=rule, random_state=random_state)
        rule.bounds[:, 1] += self.mutation(rule=rule, random_state=random_state)


