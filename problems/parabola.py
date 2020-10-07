"""
Simple implementation of the multi-dimensional frog problem.

Inspired by the one-dimensional version which can be found in Three
Architectures for Continuous Action (Wilson, 2007).
"""

from math import sqrt

import numpy as np  # type: ignore
from lcs.problems.problem import Problem


class ParabolicProblem(Problem):
    def __init__(self, seed: int, xdim: int, adim: int):
        super(ParabolicProblem, self).__init__(seed=seed, xdim=xdim, adim=adim)

        self.coef = self.random.uniform(-20, 20, (xdim + adim) * 2 + 1)

    def q(xa: np.ndarray) -> float:
        """
        Quality function for choosing a in state x.
        :param xa: array of all inputs
        :return:
        """
        return np.sum(self.coef[:-1]
                      * np.array(list(map(lambda x: np.array(
                          (x, x**2)), xa))).T.flatten()) + self.coef[-1]
