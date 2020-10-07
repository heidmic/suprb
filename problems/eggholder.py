from math import sin, sqrt

import numpy as np  # type: ignore
from lcs.problems.problem import Problem


class RegularEggcrateProblem(Problem):
    def __init__(self, seed: int, xdim: int, adim: int):
        super(RegularEggcrateProblem, self).__init__(seed=seed,
                                                     xdim=xdim,
                                                     adim=adim)
        self.coef = self.random.uniform(-1, 1, xdim + adim + 1)

    def q(xa: np.ndarray) -> float:
        """
        Quality function for choosing a in state x.
        :param xa: array of all inputs
        :return:
        """
        raise NotImplementedError("Not yet properly squashed into [-1, 1]^n!")
        return np.sum(self.coef[:-1] * np.array(list(map(
            np.sin, xa))).T.flatten()) + self.coef[-1]


class EggholderProblem(Problem):
    # NOTE: For multi-dim Eggholder, see
    # https://al-roomi.org/benchmarks/unconstrained/n-dimensions/187-egg-holder-function
    def __init__(self, seed: int):
        super(EggholderProblem, self).__init__(seed=seed, 1, 1)

    def q(xa: np.ndarray) -> float:
        """
        Quality function for choosing a in state x.
        :param xa: array of all inputs
        :return:
        """
        assert xa.shape == (2, )
        x = xa[0]
        y = xa[1]
        # squash function into [-1, 1]
        x *= 512
        y *= 512
        eggholder = -(y + 47) * sin(sqrt(abs(x / 2 + (y + 47)))) - x * sin(
            sqrt(abs(x - (y + 47))))
        return eggholder
