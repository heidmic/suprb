import numpy as np  # type: ignore
import scipy as sp  # type: ignore
from lcs.problems.gaussiansum import GaussianSumProblem
from lcs.problems.problem import Problem


class CamelProblem(Problem):
    def __init__(self, seed: int, hunches: int):
        super(CamelProblem, self).__init__(seed=seed, xdim=1, adim=1)

        # Create a positive semi-definite matrix for every hunch there should
        # be.
        PSDs = list(
            map(lambda _: GaussianSumProblem.random_psd(2, low=0, high=30),
                range(hunches)))
        shifts = list(
            map(
                lambda h: np.array(
                    [h / hunches * 2 - 1 + 1 / (hunches * 2), 0]),
                range(hunches)))
        PSDs = [np.array([[10, 5], [5, 10]]), np.array([[10, 5], [5, 10]])]
        shifts = [np.array([-0.5, 0.5]), np.array([0.5, -0.5])]

        self.gaussians = list(map(GaussianSumProblem.gaussian, PSDs, shifts))

    def q(self, xa: np.ndarray):
        return sum(map(lambda g: g(xa), self.gaussians))
