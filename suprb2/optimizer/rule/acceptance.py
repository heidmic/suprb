from abc import ABCMeta, abstractmethod

import numpy as np

from suprb2.base import BaseComponent
from suprb2.rule import Rule


class RuleAcceptance(BaseComponent, metaclass=ABCMeta):
    """Decides if a rule should be inserted into the population."""

    @abstractmethod
    def __call__(self, rule: Rule, X: np.ndarray, y: np.ndarray) -> bool:
        pass


class MaxError(RuleAcceptance):
    """Insert if the rule has an error smaller than a threshold."""

    def __init__(self, max_error: float = 0.01):
        self.max_error = max_error

    def __call__(self, rule: Rule, X: np.ndarray, y: np.ndarray) -> bool:
        return rule.error_ <= self.max_error


class Variance(RuleAcceptance):
    """
    Insert if the rule has an error smaller than the variance of matched data divided by beta.
    Note that this acceptance criterion only computes the variance in a standardized context.
    """

    def __init__(self, beta: float = 1):
        self.beta = beta

    def __call__(self, rule: Rule, X: np.ndarray, y: np.ndarray) -> bool:
        if rule.experience_ < 1:
            return False
        local_y = y[rule.match_]
        min_error = np.sum(local_y ** 2) / (len(local_y) * self.beta)
        return rule.error_ <= min_error
