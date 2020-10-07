"""
Simple implementation of the multi-dimensional frog problem.

Inspired by the one-dimensional version which can be found in Three
Architectures for Continuous Action (Wilson, 2007).
"""

from math import sqrt

import numpy as np  # type: ignore
from lcs.problems.problem import Problem


class FrogProblem(Problem):
    def __init__(self, seed: int):
        super(FrogProblem, self).__init__(seed=seed, xdim=xdim, adim=xdim)

    def q(xa: np.ndarray) -> float:
        """Quality function for choosing a in state x.

        "(…) should be any function of x and action a that is bigger the
        smaller the distance d' that remains after jumping." (Wilson, 2007)

        For now, we simply use the negative Euclidean distance between x
        and a.
        """
        x = xa[0] + 1
        a = xa[1]
        if x + a <= 1:
            return x + a
        else:
            return 2 - (x + a)
