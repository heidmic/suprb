from itertools import combinations
from math import floor, pi, sqrt
from typing import Callable

import numpy as np  # type: ignore
import scipy.stats as sp  # type: ignore
from problems.problem import Problem


class GaussianSumProblem(Problem):
    def __init__(self, seed: int, xdim: int, adim: int):
        super(GaussianSumProblem, self).__init__(seed=seed,
                                                 xdim=xdim,
                                                 adim=adim)

        perms: int = floor((xdim + adim) * (xdim + adim - 1) / 2)
        # Create positive semi-definite matrices for every combination of x's,
        # a's and x's and a's.
        PSDs = list(
            map(lambda _: self.random_psd(2, low=0, high=10), range(perms)))
        shifts = list(
            map(lambda _: self.random.uniform(low=-1, high=1, size=(2, )),
                range(perms)))

        self.gaussians = list(
            map(GaussianSumProblem.gaussian_normalized_probability, PSDs,
                shifts))
        #TODO maximum out of bounds (-1, 1). careful in paper

    def q(self, xa: np.ndarray) -> float:
        return sum(
            map(lambda g, xa2: g(np.array(xa2)), self.gaussians,
                combinations(xa, 2)))

    @classmethod
    def gaussian(cls, PSD: np.ndarray,
                 shift: np.ndarray) -> Callable[[np.ndarray], float]:
        """
        A Gaussian function based on the given positive semi-definite matrix and
        shift.

        :param shift: NumPy array of shape (2, ).
        """
        # TODO Perhaps add some negative Gaussians?
        assert PSD.shape == (shift.shape[0], shift.shape[0])

        def g(xa: np.ndarray) -> float:
            """
            :param xa: Shape (2, ).
            """
            return np.exp(-(xa - shift) @ PSD
                          @ (xa - shift)[:, np.newaxis]).reshape(())

        return g

    @classmethod
    def gaussian_normalized(
            cls, PSD: np.ndarray,
            shift: np.ndarray) -> Callable[[np.ndarray], float]:
        """
        A Gaussian function based on the given positive semi-definite matrix and
        shift with a normalized amplitude of 1.

        :param shift: NumPy array of shape (2, ).
        """
        # TODO Perhaps add some negative Gaussians?
        assert PSD.shape == (shift.shape[0], shift.shape[0])

        def g(xa: np.ndarray) -> float:
            """
            :param xa: Shape (2, ).
            """
            return np.exp(-(xa - shift) @ PSD
                          @ (xa - shift)[:, np.newaxis]).reshape(())

        amplitude = g(shift)

        def g_normalized(xa: np.ndarray) -> float:
            return g(xa) / amplitude

        return g_normalized

    @classmethod
    def gaussian_normalized_probability(
            cls, PSD: np.ndarray,
            shift: np.ndarray) -> Callable[[np.ndarray], float]:
        """
        A Gaussian function based on a multivariate normal distribution (with
        the usual normalization) with a normalized amplitude of 1 (thus, the
        result in general is not a probability density function anymore).

        :param shift: NumPy array of shape (2, ).
        """
        # TODO Perhaps add some negative Gaussians?
        assert PSD.shape == (shift.shape[0], shift.shape[0])

        k = PSD.shape[0]
        det = np.linalg.det(np.linalg.inv(PSD))

        def g(xa: np.ndarray) -> float:
            """
            :param xa: Shape (2, ).
            """
            return (np.exp(
                -0.5 *
                (xa - shift) @ PSD @ (xa - shift)[:, np.newaxis]).reshape(
                    ()) / sqrt((2 * pi)**k * det))

        amplitude = g(shift)

        def g_normalized(xa: np.ndarray) -> float:
            return g(xa) / amplitude

        return g_normalized

    def random_psd(self, dim: int, low: float, high: float) -> np.ndarray:
        """
        Generates a random positive semi-definite square matrix of shape
        (`dim`, `dim`) by (uniformly) drawing a random set of eigenvalues and a
        random set of orthogonal vectors (as eigenvectors) and then composing
        them to a matrix.

        :parameter dim: Dimensionality of the matrix.
        :parameter low: Lowest value to use in the random eigenvalues.
        :parameter high: Highest value to use in the random eigenvalues.
        """
        eigenvals = self.random.uniform(low=low, high=high, size=(dim, ))
        eigenvecs = sp.ortho_group.rvs(dim=dim)

        return eigenvecs @ np.diag(eigenvals) @ np.linalg.inv(eigenvecs)
