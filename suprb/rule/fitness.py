from abc import ABCMeta
from typing import Callable

import numpy as np

from suprb.fitness import pseudo_accuracy, actual_accuracy, emary, wu
from . import Rule, RuleFitness


class PseudoAccuracy(RuleFitness):
    """
    Use the pseudo accuracy directly as fitness.
    Computed fitness is in [0, 100].
    """

    def __call__(self, rule: Rule) -> float:
        # note that we multiply solely for readability reasons without
        # any performance impact
        if rule.isClassifier:
            return  actual_accuracy(rule.error_) * 100
        return pseudo_accuracy(rule.error_) * 100


class VolumeRuleFitness(RuleFitness, metaclass=ABCMeta):
    """Rules with bigger bounds (volume) are preferred."""

    fitness_func_: Callable

    def __init__(self, alpha: float = None):
        self.alpha = alpha

    def __call__(self, rule: Rule) -> float:
        diff = rule.input_space[:, 1] - rule.input_space[:, 0]
        diff[diff == 0] = 1.0  # avoid division by zero
        input_space_volume = np.prod(diff)

        volume_share = rule.volume_ / input_space_volume
        if rule.isClassifier:
            accuracy = actual_accuracy(rule.error_)
        else:
            accuracy = pseudo_accuracy(rule.error_, beta=2)
        return self.fitness_func_(alpha=self.alpha, x1 = accuracy, x2=volume_share) * 100


class VolumeEmary(VolumeRuleFitness):
    """
    Mixes the pseudo-accuracy as primary objective x1
    and volume as secondary objective x2 using the Emary fitness.
    Computed fitness is in [0, 100].
    """

    def __init__(self, alpha: float = 0.99):
        super().__init__(alpha)
        self.fitness_func_ = emary


class VolumeWu(VolumeRuleFitness):
    """
    Mixes the pseudo-accuracy as primary objective x1
    and volume as secondary objective x2 using the Wu fitness.
    Computed fitness is in [0, 100].
    """

    def __init__(self, alpha: float = 0.05):
        super().__init__(alpha)
        self.fitness_func_ = wu
