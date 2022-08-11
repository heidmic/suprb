from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
import math

from suprb.base import BaseComponent


class MatchingFunction(BaseComponent, metaclass=ABCMeta):
    def __call__(self, X: np.ndarray):
        """
        Determine the match set
        :param X: data matching is calculated on
        :return: a boolean array that is True for data points the rule matches
        """
        pass

    @abstractproperty
    def volume_(self):
        return None

    @abstractmethod
    def copy(self):
        """Return a deep copy"""
        pass

    @abstractmethod
    def clip(self, bounds: np.ndarray):
        """Clip a rules outer most matched examples to some value"""
        pass

    @abstractmethod
    def min_range(self, min_range: float):
        """Increase the rules matching in each dimension to avoid very small
        and long rules"""
        pass


class OrderedBound(MatchingFunction):
    """
    A standard interval-based matching function producing multi-dimensional
    hyperrectangular conditions. In effect, a lower (l) and upper bound (u) are
    specified for each dimension. Those bounds always fulfill l <= u
    An example x is matched iff l_i <= x_i <= u_i for all dimensions i
    """
    def __init__(self, bounds: np.ndarray):
        # lower_bound = bounds[:, 0]
        # upper_bound = bounds[:, 1]
        self.bounds = bounds

    def __call__(self, X: np.ndarray):
        return np.all((self.bounds[:, 0] <= X) & (X <= self.bounds[:, 1]), axis=1)

    @property
    def volume_(self):
        """Calculates the volume of the interval."""
        diff = self.bounds[:, 1] - self.bounds[:, 0]
        return np.prod(diff)

    def copy(self):
        return OrderedBound(self.bounds.copy())

    def _validate_bounds(self, X: np.ndarray):
        """Validates that bounds have the correct shape."""

        if self.bounds.shape[1] != 2:
            raise ValueError(f"specified bounds are not of shape (-1, 2), but {self.bounds.shape}")

        if self.bounds.shape[0] != X.shape[1]:
            raise ValueError(f"bounds- and input data dimension mismatch: {self.bounds.shape[0]} != {X.shape[1]}")

    def clip(self, bounds: np.ndarray):
        low, high = bounds[None].T
        bounds.clip(low, high)

    def min_range(self, min_range: float):
        diff = self.bounds[:, 1] - self.bounds[:, 0]
        if min_range > 0:
            invalid_indices = np.argwhere(diff < min_range)

            self.bounds[invalid_indices, 0] -= min_range / 2
            self.bounds[invalid_indices, 1] += min_range / 2


class GaussianKernelFunction(MatchingFunction):
    """
    A standard kernel-based matching function producing multi-dimensional
    hyperellipsoidal conditions. In effect, a center point (c) and deviations (d)
    are specified for each dimension. A threshhold (t) defines if a rule is matched or not.
    An example x is matched iff exp((x_i - c_i)^2 / 2*(d_i^2)) > t for all dimensions i
    """
    def __init__(self, bounds: np.ndarray, threshold: float = 0.7):
        # center = bounds[:, 0]
        # deviations = bounds[:, 1]
        self.bounds = bounds
        self.threshold = threshold

    def __call__(self, X: np.ndarray):
        return np.exp(np.sum(
                ((X - self.bounds[:, 0]) ** 2) / (2 * (self.bounds[:, 1] ** 2)), axis=1) * -1) > self.threshold

    @property
    def volume_(self):
        """
        Calculates the volume of the ellipsoid.
        Reference how to calculate the volume of an n-dim ellipsoid:
        https://analyticphysics.com/Higher%20Dimensions/Ellipsoids%20in%20Higher%20Dimensions.htm
        """

        dim = self.bounds[:, 0].shape[0]
        pre_factor = (2 * (np.pi ** (dim / 2))) / (dim * math.gamma(dim / 2))
        prod_deviations = np.prod(self.bounds[:, 1])

        return pre_factor * prod_deviations

    def copy(self):
        return GaussianKernelFunction(bounds=self.bounds, threshold=self.threshold)

    def _validate_bounds(self, X: np.ndarray):
        """Validates that bounds have the correct shape."""

        if self.bounds.shape[1] != 2:
            raise ValueError(f"specified bounds are not of shape (-1, 2), but {self.bounds.shape}")

        if self.bounds.shape[0] != X.shape[1]:
            raise ValueError(f"bounds- and input data dimension mismatch: {self.bounds.shape[0]} != {X.shape[1]}")

    def clip(self, bounds: np.ndarray):
        # TODO: Noch abzuklären was hier genau gemacht werden soll -> ob dann die implementierung passt
        low, high = bounds[:, 0] - bounds[:, 1], bounds[:, 0] + bounds[:, 1]
        radius = np.abs(high - low) / 2

        bounds[:, 0] = bounds[:, 0].clip(low, high)[0, :]
        bounds[:, 1] = bounds[:, 1].clip(0, radius)[0, :]

    def min_range(self, min_range: float):
        # TODO: Noch abzuklären was hier genau gemacht werden soll -> ob dann die implementierung passt
        # beschreibt min_range den Durchmesser oder Radius?
        low, high = self.bounds[:, 0] - self.bounds[:, 1], self.bounds[:, 0] + self.bounds[:, 1]

        #low = low.clip(-1, 1)
        #high = high.clip(-1, 1)

        diameter = high - low

        if min_range > 0:
            invalid_indices = np.argwhere(diameter < min_range)
            self.bounds[invalid_indices, 1] += min_range / 2