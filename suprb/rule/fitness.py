from abc import ABCMeta
from typing import Callable

import numpy as np
import warnings

from suprb.fitness import pseudo_accuracy, emary, wu, moo
from . import Rule, RuleFitness
from ..fitness import BaseFitness


class PseudoAccuracy(RuleFitness):
    """
    Use the pseudo accuracy directly as fitness.
    Computed fitness is in [0, 100].
    """

    def __call__(self, rule: Rule) -> float:
        # note that we multiply solely for readability reasons without
        # any performance impact
        return pseudo_accuracy(rule.error_) * 100


class VolumeRuleFitness(RuleFitness, metaclass=ABCMeta):
    """Rules with bigger bounds (volume) are preferred."""

    fitness_func_: Callable

    def __init__(self, alpha: float = None):
        self.alpha = alpha

    def __call__(self, rule: Rule) -> float:
        diff = rule.input_space[:, 1] - rule.input_space[:, 0]
        input_space_volume = np.prod(diff)

        volume_share = rule.volume_ / input_space_volume
        return (
            self.fitness_func_(
                alpha=self.alpha,
                x1=pseudo_accuracy(rule.error_, beta=2),
                x2=volume_share,
            )
            * 100
        )


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

class MooFitness(RuleFitness, metaclass=ABCMeta):
    """
    A placeholder fitness function which always returns NaN.
    This function is used in multi-objective optimization (MOO) as the rule attribute `.fitness_` is never accessed.
    Using another fitness function has no impact on the MOO and may be misleading.

    Changing the fitness attribute to a vector may make the code more intuitive.
    """

    fitness_func_: Callable

    def __init__(self):
        super().__init__()
        self.fitness_func_ = moo
        warnings.warn(
            "MooFitness is a dummy fitness function which always returns NaN. "
            "When using multi-objective optimization (MOO), the rule attribute `.fitness_` is never accessed. "
            "Using another fitness function has no impact on the MOO and may be misleading.",
            UserWarning,
            stacklevel=2,
        )

    def __call__(self, rule: Rule) -> float:
        return self.fitness_func_()