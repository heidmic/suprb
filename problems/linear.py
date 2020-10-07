"""
Simple implementation of the multi-dimensional frog problem.

Inspired by the one-dimensional version which can be found in Three
Architectures for Continuous Action (Wilson, 2007).
"""

from math import sqrt

import numpy as np  # type: ignore
from lcs.problems.problem import Problem


class LinearProblem(Problem):
    def __init__(self, seed: int, xdim: int, a_dim: int):
        super(LinearProblem, self).__init__(seed=seed, xdim=xdim, adim=xdim)
        self.coef = self.random.uniform(-10, 10, (xdim + a_dim) + 1)

    def q(xa: np.ndarray) -> float:
        """
        Quality function for choosing a in state x.
        :param xa: array of all inputs
        :return:
        """
        return np.sum(self.coef[0:-1] * xa) + self.coef[-1]
