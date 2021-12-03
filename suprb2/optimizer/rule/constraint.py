from __future__ import annotations

from abc import ABCMeta, abstractmethod

import numpy as np

from suprb2.base import BaseComponent
from suprb2.rule import Rule


class RuleConstraint(BaseComponent, metaclass=ABCMeta):
    """
    Represents a constraint that rules have to fulfill, and changes the rule in order to fit this constraint.
    Note that the rule is mutated directly, and not copied.
    Returns the constrained rule for convenience.
    """

    @abstractmethod
    def __call__(self, rule: Rule) -> Rule:
        pass


class CombinedConstraint(RuleConstraint):
    """
    Apply min_range and clip to a rule.
    Could be extended to arbitrary constraints, but it is too tiresome for now to
    implement the get_params() and set_params() methods for many constraints.
    """

    def __init__(self, min_range: MinRange, clip: Clip):
        self.min_range = min_range
        self.clip = clip

    def __call__(self, rule: Rule) -> Rule:
        return self.clip(self.min_range(rule))


class Clip(RuleConstraint):
    """Clip the rule into bounds."""

    def __init__(self, bounds: np.ndarray = None):
        self.bounds = bounds

    def __call__(self, rule: Rule) -> Rule:
        low, high = self.bounds[None].T
        rule.bounds = rule.bounds.clip(low, high)
        return rule


class MinRange(RuleConstraint):
    """Make bounds bigger that were generated smaller than min_range."""

    def __init__(self, min_range: float = 1e-6):
        self.min_range = min_range

    def __call__(self, rule: Rule) -> Rule:
        diff = rule.bounds[:, 1] - rule.bounds[:, 0]
        if self.min_range > 0:
            invalid_indices = np.argwhere(diff < self.min_range)
            rule.bounds[invalid_indices, 0] -= self.min_range / 2
            rule.bounds[invalid_indices, 1] += self.min_range / 2
        return rule
