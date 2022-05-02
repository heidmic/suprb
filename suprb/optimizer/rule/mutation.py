from abc import ABCMeta

import numpy as np
from scipy.stats import halfnorm

from suprb.base import BaseComponent
from suprb.rule import Rule
from suprb.utils import RandomState


class RuleMutation(BaseComponent, metaclass=ABCMeta):
    """Mutates the bounds of a rule with the strength defined by sigma."""

    def __init__(self, sigma_lower: float = 0.1, sigma_prop: float = 0.01):
        self.sigma_lower = sigma_lower
        self.sigma_prop = sigma_prop

    def __call__(self, rule: Rule, random_state: RandomState) -> Rule:
        # Create copy of the rule
        mutated_rule = rule.clone()

        # Mutation
        self.mutate_bounds(mutated_rule, random_state)
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
        self.mutation.sigma_lower = self.sigma
        return self.mutation(rule, random_state)


class Normal(RuleMutation):
    """Normal noise on both lower bound and proportion
    weighed with a factor of sigma_lower and sigma_prop respectively."""

    def mutate_bounds(self, rule: Rule, random_state: RandomState):
        rule.bounds[:, 0] += random_state.normal(scale=self.sigma_lower, size=rule.bounds.shape[0])
        rule.bounds[:, 1] += random_state.normal(scale=self.sigma_prop, size=rule.bounds.shape[0])


class HalfnormIncrease(RuleMutation):
    """Increases the distance proportion with
    (half)normal noise and decreases the lower bound with (half)normal noise."""

    def mutation(self, rule: Rule, random_state: RandomState):
        return halfnorm.rvs(scale=self.sigma_prop / 2, size=rule.bounds.shape[0], random_state=random_state)

    def mutate_bounds(self, rule: Rule, random_state: RandomState):
        rule.bounds[:, 1] += self.mutation(rule=rule, random_state=random_state)
        rule.bounds[:, 0] -= \
            halfnorm.rvs(scale=self.sigma_lower / 2, size=rule.bounds.shape[0], random_state=random_state)


class UniformIncrease(RuleMutation):
    """Increase the distances proportion with uniform noise and decreases the lower bound using normal noise."""

    def mutation(self, rule: Rule, random_state: RandomState):
        return random_state.uniform(0, self.sigma_prop, size=rule.bounds.shape[0])

    def mutate_bounds(self, rule: Rule, random_state: RandomState):
        rule.bounds[:, 1] += self.mutation(rule=rule, random_state=random_state)
        rule.bounds[:, 0] -= \
            random_state.uniform(0, self.sigma_lower, size=rule.bounds.shape[0])