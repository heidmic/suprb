from abc import abstractmethod, ABCMeta

import numpy as np

from suprb2.optimizer.fitness import BaseFitness
from suprb2.optimizer.fitness import pseudo_accuracy, emary, wu
from suprb2.rule import Rule


class RuleFitness(BaseFitness, metaclass=ABCMeta):
    """Evaluates the fitness of a `Rule`."""

    @abstractmethod
    def __call__(self, rule: Rule) -> float:
        pass


class PseudoAccuracy(RuleFitness):
    """
    Use the pseudo accuracy directly as fitness.
    Computed fitness is in [0, 100].
    """

    def __call__(self, rule: Rule) -> float:
        return pseudo_accuracy(rule.error_) * 100


class VolumeRuleFitness(RuleFitness, metaclass=ABCMeta):
    """Rules with bigger bounds (volume) are preferred."""

    def __init__(self, alpha: float, bounds: np.ndarray):
        self.alpha = alpha
        self.bounds = bounds

    @property
    def bounds_volume_(self):
        diff = self.bounds[:, 1] - self.bounds[:, 0]
        return np.prod(diff)


class VolumeEmary(VolumeRuleFitness):
    """
    Mixes the pseudo-accuracy as primary objective x1
    and volume as secondary objective x2 using the Emary fitness.
    Computed fitness is in [0, 100].
    """

    def __init__(self, alpha: float = 0.85, bounds: np.ndarray = None):
        super().__init__(alpha, bounds)

    def __call__(self, rule: Rule) -> float:
        volume_share = rule.volume_ / self.bounds_volume_
        return emary(self.alpha, pseudo_accuracy(rule.error_, beta=2), volume_share) * 100


class VolumeWu(VolumeRuleFitness):
    """
    Mixes the pseudo-accuracy as primary objective x1
    and volume as secondary objective x2 using the Wu fitness.
    Computed fitness is in [0, 100].
    """

    def __init__(self, alpha: float = 0.05, bounds: np.ndarray = None):
        super().__init__(alpha, bounds)

    def __call__(self, rule: Rule) -> float:
        volume_share = rule.volume_ / self.bounds_volume_
        return wu(self.alpha, pseudo_accuracy(rule.error_, beta=2), volume_share) * 100
