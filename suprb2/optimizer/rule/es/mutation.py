from abc import ABCMeta
from copy import deepcopy

import numpy as np
from scipy.stats import halfnorm
from sklearn import clone

from suprb2.base import BaseComponent
from suprb2.rule import Rule


class RuleMutation(BaseComponent, metaclass=ABCMeta):
    """Mutates the bounds of a rule with the strength defined by sigma."""

    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def __call__(self, rule: Rule, random_state: np.random.RandomState) -> Rule:
        # Create copy of the rule
        mutation = Rule(deepcopy(rule.bounds), clone(rule.model))
        self.mutate_bounds(mutation, random_state)
        mutation.bounds = np.sort(mutation.bounds, axis=1)
        return mutation

    def mutate_bounds(self, rule: Rule, random_state: np.random.RandomState):
        pass


class SigmaRange(RuleMutation):
    """Draws the sigma used for another mutation from uniform distribution, low to high."""

    def __init__(self, mutation: RuleMutation, low: float, high: float):
        super().__init__(0)
        self.mutation = mutation
        self.low = low
        self.high = high

    def __call__(self, rule: Rule, random_state: np.random.RandomState) -> Rule:
        self.sigma = random_state.uniform(self.low, self.high)
        self.mutation.sigma = self.sigma
        return self.mutation(rule, random_state)


class Normal(RuleMutation):
    """Normal noise on both bounds."""

    def mutate_bounds(self, rule: Rule, random_state: np.random.RandomState):
        rule.bounds += random_state.normal(scale=self.sigma, size=rule.bounds.shape)


class Halfnorm(RuleMutation):
    """Sample with (half)normal distribution around mean."""
    type_ = 'halfnorm'

    def mutate_bounds(self, rule: Rule, random_state: np.random.RandomState):
        mean = np.mean(rule.bounds, axis=1)
        rule.bounds[:, 0] = mean - halfnorm.rvs(scale=self.sigma / 2, size=rule.bounds.shape[0],
                                                random_state=random_state)
        rule.bounds[:, 1] = mean + halfnorm.rvs(scale=self.sigma / 2, size=rule.bounds.shape[0],
                                                random_state=random_state)


class HalfnormIncrease(RuleMutation):
    """Increase bounds with (half)normal noise."""

    def mutate_bounds(self, rule: Rule, random_state: np.random.RandomState):
        rule.bounds[:, 0] -= halfnorm.rvs(scale=self.sigma / 2, size=rule.bounds.shape[0], random_state=random_state)
        rule.bounds[:, 1] += halfnorm.rvs(scale=self.sigma / 2, size=rule.bounds.shape[0], random_state=random_state)


class Uniform(RuleMutation):
    """Uniform noise on both bounds."""

    def mutate_bounds(self, rule: Rule, random_state: np.random.RandomState):
        rule.bounds += random_state.uniform(-self.sigma, self.sigma, size=rule.bounds.shape)


class UniformIncrease(RuleMutation):
    """Increase bounds with uniform noise."""

    def mutate_bounds(self, rule: Rule, random_state: np.random.RandomState):
        rule.bounds[:, 0] += random_state.uniform(-self.sigma, 0, size=rule.bounds.shape[0])
        rule.bounds[:, 1] += random_state.uniform(0, self.sigma, size=rule.bounds.shape[0])
