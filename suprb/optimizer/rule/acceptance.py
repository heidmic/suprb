from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score

from suprb.base import BaseComponent
from suprb.rule import Rule
from suprb.fitness import pseudo_error


class RuleAcceptance(BaseComponent, metaclass=ABCMeta):
    """Decides if a rule should be inserted into the population."""

    @abstractmethod
    def __call__(self, rule: Rule, X: np.ndarray, y: np.ndarray) -> bool:
        pass


class MaxError(RuleAcceptance):
    """Insert if the rule has an error smaller or equal to a threshold."""

    def __init__(self, max_error: float = 0.01):
        self.max_error = max_error

    def __call__(self, rule: Rule, X: np.ndarray, y: np.ndarray) -> bool:
        return rule.error_ <= self.max_error


class Variance(RuleAcceptance):
    """
    Insert if the rule has an error smaller or equal to the variance of matched data divided by beta.
    Note that this acceptance criterion only computes the variance in a standardized context.
    """

    def __init__(self, beta: float = 1):
        self.beta = beta

    def __call__(self, rule: Rule, X: np.ndarray, y: np.ndarray) -> bool:
        if rule.experience_ < 1:
            return False
        local_y = y[rule.match_set_]
        default_error = np.sum(local_y ** 2) / (len(local_y) * self.beta)
        if rule.isClassifier:
            # default error is the trivial solution of always choosing the most common label
            local_y = [round(y) for y in local_y]
            default_accuracy = np.bincount(local_y).max() / (len(local_y) * self.beta)
            default_error = pseudo_error(default_accuracy)
        return rule.error_ <= default_error


class Precision(RuleAcceptance):
    """Insert if the rule has a precision greater or equal to a threshold"""

    def __init__(self, min_precission: float = 0.9):
        self.min_precission = min_precission

    def __call__(self, rule: Rule, X: np.ndarray, y: np.ndarray) -> bool:
        if rule.experience_ < 1:
            return False
        local_y = y[rule.match_set_]
        return precision_score(local_y, rule.predict(X[rule.match_set_]), average='macro', zero_division=np.nan) >= self.min_precission
    
class F1_Score(RuleAcceptance):
    """Insert if the rule has a f1_score greater or equal to a threshold"""

    def __init__(self, min_f1: float = 0.9):
        self.min_precission = min_f1

    def __call__(self, rule: Rule, X: np.ndarray, y: np.ndarray) -> bool:
        if rule.experience_ < 1:
            return False
        local_y = y[rule.match_set_]
        return f1_score(local_y, rule.predict(X[rule.match_set_])) >= self.min_f1