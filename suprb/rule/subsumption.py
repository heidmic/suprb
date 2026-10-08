from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np

from sklearn.metrics import mean_squared_error

from suprb.base import BaseComponent
from .base import Rule
from .matching import OrderedBound, UnorderedBound, CenterSpread, MinPercentage


def get_effective_bounds(match) -> np.ndarray:
    """Returns (n_features, 2) array of [lower, upper], regardless of the
    underlying MatchingFunction representation."""
    if isinstance(match, OrderedBound):
        return match.bounds
    if isinstance(match, UnorderedBound):
        lower = np.min(match.bounds, axis=1)
        upper = np.max(match.bounds, axis=1)
        return np.stack([lower, upper], axis=1)
    if isinstance(match, CenterSpread):
        lower = match.bounds[:, 0] - match.bounds[:, 1]
        upper = match.bounds[:, 0] + match.bounds[:, 1]
        return np.stack([lower, upper], axis=1)
    if isinstance(match, MinPercentage):
        lower = match.bounds[:, 0]
        upper = lower + match.bounds[:, 1] * (1 - lower)
        return np.stack([lower, upper], axis=1)
    raise NotImplementedError(f"Unhandled matching function type: {type(match)}")


def bounds_contains(outer: np.ndarray, inner: np.ndarray) -> bool:
    """True if `outer` geometrically contains `inner` on every dimension."""
    return bool(np.all(outer[:, 0] <= inner[:, 0]) and np.all(outer[:, 1] >= inner[:, 1]))


def local_error_on_subset(containing_rule: Rule, X: np.ndarray, y: np.ndarray, subset_mask: np.ndarray) -> float:

    X_sub, y_sub = X[subset_mask], y[subset_mask]
    pred_sub = containing_rule.predict(X_sub)
    return max(mean_squared_error(y_sub, pred_sub), 1e-4)


class RuleSubsumption(BaseComponent, metaclass=ABCMeta):
    """
    Decides, for a newly generated rule, whether it subsumes or is subsumed
    by rules already present in the pool.

      (a) discards the new rule entirely, if an existing pool rule already
          subsumes it, or
      (b) overwrites a single existing pool slot in place (same index, same
          list length) with the new rule, if the new rule subsumes it, or
      (c) leaves the new rule to be appended normally, if neither applies.
    """

    tolerance: float = 0.0

    @abstractmethod
    def resolve_mutual(self, rule_a: Rule, rule_b: Rule) -> tuple[Rule, Rule]:
        """Given two mutually-subsuming rules, return (winner, loser).
        `loser`'s numerosity is folded into `winner`."""
        pass

    def __call__(self, new_rule: Rule, pool: list[Rule], X: np.ndarray, y: np.ndarray) -> tuple[bool, Optional[int]]:
        new_bounds = get_effective_bounds(new_rule.match)

        # --- Pass 1: does any existing pool rule dominate new_rule outright?
        for pool_rule in pool:
            pool_bounds = get_effective_bounds(pool_rule.match)

            pool_contains_new = bounds_contains(pool_bounds, new_bounds)
            if not pool_contains_new:
                continue

            pool_local_error = local_error_on_subset(pool_rule, X, y, new_rule.match_set_)
            pool_at_least_as_accurate = pool_local_error <= new_rule.error_ * (1 + self.tolerance)
            if not pool_at_least_as_accurate:
                continue

            new_contains_pool = bounds_contains(new_bounds, pool_bounds)
            new_local_error = local_error_on_subset(new_rule, X, y, pool_rule.match_set_)
            new_at_least_as_accurate = new_local_error <= pool_rule.error_ * (1 + self.tolerance)
            mutual = new_contains_pool and new_at_least_as_accurate

            if mutual:
                winner, loser = self.resolve_mutual(new_rule, pool_rule)
                if winner is pool_rule:
                    pool_rule.numerosity_ += new_rule.numerosity_
                    return False, None
                else:
                    continue
            else:
                pool_rule.numerosity_ += new_rule.numerosity_
                return False, None

        # --- Pass 2: does new_rule dominate any existing pool rule(s)?
        replace_idx: Optional[int] = None
        for idx, pool_rule in enumerate(pool):
            pool_bounds = get_effective_bounds(pool_rule.match)

            new_contains_pool = bounds_contains(new_bounds, pool_bounds)
            if not new_contains_pool:
                continue

            new_local_error = local_error_on_subset(new_rule, X, y, pool_rule.match_set_)
            new_at_least_as_accurate = new_local_error <= pool_rule.error_ * (1 + self.tolerance)
            if not new_at_least_as_accurate:
                continue

            new_rule.numerosity_ += pool_rule.numerosity_
            if replace_idx is None:
                replace_idx = idx

        return True, replace_idx


class PreferSmallerVolume(RuleSubsumption):
    """Tie-break for mutual subsumption: keep the smaller-volume rule."""

    def __init__(self, tolerance: float = 0.0):
        self.tolerance = tolerance

    def resolve_mutual(self, rule_a: Rule, rule_b: Rule) -> tuple[Rule, Rule]:
        return (rule_a, rule_b) if rule_a.volume_ <= rule_b.volume_ else (rule_b, rule_a)


class PreferLargerVolume(RuleSubsumption):
    """Tie-break for mutual subsumption: keep the larger-volume rule."""

    def __init__(self, tolerance: float = 0.0):
        self.tolerance = tolerance

    def resolve_mutual(self, rule_a: Rule, rule_b: Rule) -> tuple[Rule, Rule]:
        return (rule_a, rule_b) if rule_a.volume_ >= rule_b.volume_ else (rule_b, rule_a)
