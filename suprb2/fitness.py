from abc import ABCMeta, abstractmethod

import numpy as np

from suprb2.base import BaseComponent, Solution


class BaseFitness(BaseComponent, metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, solution: Solution) -> float:
        pass


def pseudo_accuracy(error: float, beta=1) -> float:
    """
    Calculates the pseudo accuracy PACC, which maps the (possibly unbounded) error into the [0, 1] domain with
    a PACC of 1 corresponding to an error of 0.
    """
    assert beta > 0
    return np.exp(-beta * error)


def emary(alpha: float, x1: float, x2: float) -> float:
    """
    A fitness function which mixes two objectives x1, x2 weighted by alpha.
    Computed fitness is in [0, 1], if x1 and x2 are.
    Taken from https://doi.org/10.1016/j.neucom.2015.06.083.
    """

    return alpha * x1 + (1 - alpha) * x2


def wu(alpha: float, x1: float, x2: float) -> float:
    """
    A fitness function which mixes two objectives x1, x2 weighted by alpha.
    Computed fitness is in [0, 1], if x1 and x2 are.
    Taken from https://doi.org/10.1109/IJCNN.2018.8489676.
    """

    return ((1 + alpha ** 2) * x1 * x2) / (alpha ** 2 * x1 + x2)
