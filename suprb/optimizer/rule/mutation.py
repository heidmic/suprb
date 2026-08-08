from typing import Union

import numpy as np
from scipy.stats import halfnorm
import warnings

from suprb.rule import Rule
from suprb.utils import RandomState
from suprb.optimizer.rule.generation_operator import GenerationOperator
from suprb.rule.matching import MatchingFunction


class RuleMutation(GenerationOperator):
    """Mutates the bounds of a rule with the strength defined by sigma."""

    def __init__(
        self,
        matching_type: MatchingFunction = None,
        sigma: Union[float, np.ndarray] = 0.1,
    ):
        super().__init__(matching_type=matching_type)
        self.sigma = sigma

    def __call__(self, rule: Rule, random_state: RandomState) -> Rule:
        # Create copy of the rule
        mutated_rule = rule.clone()

        # Mutation
        self.execute(mutated_rule, random_state)

        return mutated_rule

    def execute(self, rule: Rule, random_state: RandomState):
        warnings.warn("No matching_type was set! This will impact mutation.")
        pass


class SigmaRange(RuleMutation):
    """Draws the sigma used for another mutation from uniform distribution, low to high.
    unordered_bound, ordered_bound, center_spread and min_percentage are empty, because
    of the inheritance to RuleMutation (and then the inheritance to GenerationOperator),
    which need an implementation for those functions.
    They provide no utility, since the RuleMutation this class uses is set in the __init__
    """

    def __init__(self, mutation: RuleMutation = None, low: float = 0.001, high: float = 0.1):
        super().__init__(0)
        self.mutation = mutation
        self.low = low
        self.high = high

    def __call__(self, rule: Rule, random_state: RandomState) -> Rule:
        uniform_size = None if isinstance(self.sigma, float) else len(self.sigma)
        self.sigma = random_state.uniform(self.low, self.high, uniform_size)
        self.mutation.sigma = self.sigma
        return self.mutation(rule, random_state)

    def unordered_bound(self, rule: Rule, random_state: RandomState):
        pass

    def ordered_bound(self, rule: Rule, random_state: RandomState):
        pass

    def center_spread(self, rule: Rule, random_state: RandomState):
        pass

    def min_percentage(self, rule: Rule, random_state: RandomState):
        pass
    def gabil(self, rule: Rule, random_state: RandomState):
        raise TypeError("This mutation is not implemented for GABIL")


class Normal(RuleMutation):
    """Normal noise on both bounds."""

    def individual_mutate(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        for index in range(0, self.sigma.shape[0]):
            bounds[:, index] += random_state.normal(scale=self.sigma[index], size=rule.match.bounds.shape[0])

    def unordered_bound(self, rule: Rule, random_state: RandomState):
        rule.match.bounds += random_state.normal(scale=self.sigma, size=rule.match.bounds.shape)

    def ordered_bound(self, rule: Rule, random_state: RandomState):
        # code inspection gives you a warning here but it is ineffectual
        self.unordered_bound(rule, random_state)
        rule.match.bounds = np.sort(rule.match.bounds, axis=1)

    def center_spread(self, rule: Rule, random_state: RandomState):
        self.individual_mutate(rule, random_state)

    def min_percentage(self, rule: Rule, random_state: RandomState):
        self.individual_mutate(rule, random_state)
    def gabil(self, rule: Rule, random_state: RandomState):
        raise TypeError("This mutation is not implemented for GABIL")


class Halfnorm(RuleMutation):
    """Sample with (half)normal distribution around the center."""

    def unordered_bound(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        mean = np.mean(bounds, axis=1)
        bounds[:, 0] = mean - halfnorm.rvs(scale=self.sigma / 2, size=bounds.shape[0], random_state=random_state)

        bounds[:, 1] = mean + halfnorm.rvs(scale=self.sigma / 2, size=bounds.shape[0], random_state=random_state)

    def ordered_bound(self, rule: Rule, random_state: RandomState):
        self.unordered_bound(rule, random_state)
        rule.match.bounds = np.sort(rule.match.bounds, axis=1)

    def center_spread(self, rule: Rule, random_state: RandomState):
        raise TypeError("Halform Mutation is not implemented for CSR")

    def min_percentage(self, rule: Rule, random_state: RandomState):
        raise TypeError("Halform Mutation is not implemented for MPR")
    def gabil(self, rule: Rule, random_state: RandomState):
        raise TypeError("This mutation is not implemented for GABIL")


class HalfnormIncrease(RuleMutation):
    """Increase bounds with (half)normal noise."""

    def unordered_bound(self, rule: Rule, random_state: RandomState):
        raise TypeError("HalformIncrease would cause UBR to behave like OBR")

    def ordered_bound(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        bounds[:, 0] -= halfnorm.rvs(scale=self.sigma / 2, size=bounds.shape[0], random_state=random_state)
        bounds[:, 1] += halfnorm.rvs(scale=self.sigma / 2, size=bounds.shape[0], random_state=random_state)
        rule.match.bounds = np.sort(rule.match.bounds, axis=1)

    def center_spread(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        bounds[:, 0] += random_state.normal(scale=self.sigma[0], size=bounds.shape[0])
        bounds[:, 1] += halfnorm.rvs(scale=self.sigma[1] / 2, size=bounds.shape[0], random_state=random_state)

    def min_percentage(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        bounds[:, 0] -= halfnorm.rvs(scale=self.sigma[0] / 2, size=bounds.shape[0], random_state=random_state)
        bounds[:, 1] += halfnorm.rvs(scale=self.sigma[1] / 2, size=bounds.shape[0], random_state=random_state)
    def gabil(self, rule: Rule, random_state: RandomState):
        """Adding Alternative: switch further category bits on.

        The interval variant of this operator only ever widens a rule,
        because covering starts from a zero-width interval that has to be
        generalised. GABIL's counterpart is the "adding alternative"
        operator of De Jong & Spears (1991): flip bits from 0 to 1 only,
        never the reverse. Covering sets exactly one bit, so the same
        generalisation pressure applies here.

        `sigma` is reused as the per-bit flip probability, so the existing
        mutation-strength parameter keeps its meaning without adding a new
        hyperparameter for the categorical case.
        """
        bits = rule.match.bits
        if bits.size == 0:
            return

        rate = float(np.asarray(self.sigma).ravel()[0])

        # Only unset bits are candidates, which is what makes this
        # generalisation-only rather than an unbiased bit flip.
        candidates = ~bits
        flips = candidates & (random_state.random(bits.size) < rate)
        bits[flips] = True


class Uniform(RuleMutation):
    """Uniform noise on both bounds."""

    def individual_mutate(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        for index in range(0, self.sigma.shape[0]):
            bounds[:, index] += random_state.uniform(-self.sigma[index], self.sigma[index], size=bounds.shape[0])

    def unordered_bound(self, rule: Rule, random_state: RandomState):
        rule.match.bounds += random_state.uniform(-self.sigma, self.sigma, size=rule.match.bounds.shape)

    def ordered_bound(self, rule: Rule, random_state: RandomState):
        self.unordered_bound(rule, random_state)
        rule.match.bounds = np.sort(rule.match.bounds, axis=1)

    def center_spread(self, rule: Rule, random_state: RandomState):
        self.individual_mutate(rule, random_state)

    def min_percentage(self, rule: Rule, random_state: RandomState):
        self.individual_mutate(rule, random_state)
    def gabil(self, rule: Rule, random_state: RandomState):
        raise TypeError("This mutation is not implemented for GABIL")


class UniformIncrease(RuleMutation):
    """Increase bounds with uniform noise."""

    def unordered_bound(self, rule: Rule, random_state: RandomState):
        raise TypeError("UniformIncrease would cause UBR to behave like OBR")

    def ordered_bound(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        bounds[:, 0] -= random_state.uniform(0, self.sigma, size=bounds.shape[0])
        bounds[:, 1] += random_state.uniform(0, self.sigma, size=bounds.shape[0])
        rule.match.bounds = np.sort(rule.match.bounds, axis=1)

    def center_spread(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        bounds[:, 0] -= random_state.uniform(-self.sigma[0], self.sigma[0], size=bounds.shape[0])
        bounds[:, 1] += random_state.uniform(0, self.sigma[1], size=bounds.shape[0])

    def min_percentage(self, rule: Rule, random_state: RandomState):
        bounds = rule.match.bounds
        bounds[:, 0] -= random_state.uniform(0, self.sigma[0], size=bounds.shape[0])
        bounds[:, 1] += random_state.uniform(0, self.sigma[1], size=bounds.shape[0])
    def gabil(self, rule: Rule, random_state: RandomState):
        raise TypeError("This mutation is not implemented for GABIL")
