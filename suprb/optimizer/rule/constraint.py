from __future__ import annotations

from abc import ABCMeta, abstractmethod

import numpy as np

from suprb.base import BaseComponent
from suprb.rule import Rule


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
    """Clip the center into bounds and the spread into [0,diff of bounds] """

    def __init__(self, bounds: np.ndarray = None):
        # if not set here it is set during SupRB._init_bounds()
        self.bounds = bounds

    def __call__(self, rule: Rule) -> Rule:
        low, high = self.bounds[None].T
        diff = np.abs(high - low)
        # Indexing needed because clipping creates 3 - dimensional array
        rule.bounds[:, 0] = rule.bounds[:, 0].clip(low, high)[0, :]
        rule.bounds[:, 1] = rule.bounds[:, 1].clip(0, diff)[0, :]
        return rule


class MinRange(RuleConstraint):
    """Calculate bounds and clip them into range, then increase spread for bounds with a
    diff lower than min_range."""

    def __init__(self, min_range: float = 1e-6):
        self.min_range = min_range

    def __call__(self, rule: Rule) -> Rule:
        low = rule.bounds[:, 0] - rule.bounds[:, 1]
        high = rule.bounds[:, 0] + rule.bounds[:, 1]
        low = low.clip(-1, 1)
        high = high.clip(-1, 1)
        diff = high - low
        if self.min_range > 0:
            invalid_indices = np.argwhere(diff < self.min_range)
            rule.bounds[invalid_indices, 0] -= self.min_range / 2
            rule.bounds[invalid_indices, 1] += self.min_range / 2
        return rule
