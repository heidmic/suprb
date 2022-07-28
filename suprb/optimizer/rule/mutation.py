from abc import ABCMeta

import numpy as np
from scipy.stats import halfnorm

from suprb.base import BaseComponent
from suprb.rule import Rule
from suprb.utils import RandomState


class RuleMutation(BaseComponent, metaclass=ABCMeta):
    """Mutates the bounds of a rule with the strength defined by sigma."""

    def __init__(self, sigma_center: float = 0.1, sigma_deviations: float = 0.01):
        self.sigma_center = sigma_center
        self.sigma_deviations = sigma_deviations

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
        self.mutation.sigma_center = self.sigma
        return self.mutation(rule, random_state)


class Normal(RuleMutation):
    """Normal noise on both center and spread using the respective scale."""

    def mutate_bounds(self, rule: Rule, random_state: RandomState):
        rule.bounds[:, 0] += random_state.normal(scale=self.sigma_center, size=rule.bounds.shape[0])
        rule.bounds[:, 1] += random_state.normal(scale=self.sigma_deviations, size=rule.bounds.shape[0])


class HalfnormIncrease(RuleMutation):
    """Increases the spread with (half)normal noise and moves the center with normal noise."""

    def mutation(self, rule: Rule, random_state: RandomState):
        return halfnorm.rvs(scale=self.sigma_deviations / 2, size=rule.bounds.shape[0], random_state=random_state)

    def mutate_bounds(self, rule: Rule, random_state: RandomState):
        rule.bounds[:, 0] += random_state.normal(scale=self.sigma_center, size=rule.bounds.shape[0])
        rule.bounds[:, 1] += self.mutation(rule=rule, random_state=random_state)


class UniformIncrease(RuleMutation):
    """Increase the deviations in each dimension with uniform noise and moves the center with normal noise."""

    def mutation(self, rule: Rule, random_state: RandomState):
        return random_state.uniform(0, self.sigma_deviations, size=rule.bounds.shape[0])

    def mutate_bounds(self, rule: Rule, random_state: RandomState):
        rule.bounds[:, 0] += random_state.normal(scale=self.sigma_center, size=rule.bounds.shape[0])
        rule.bounds[:, 1] += self.mutation(rule=rule, random_state=random_state)
