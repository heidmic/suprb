from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np

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

    def _validate_bounds(self, X: np.ndarray):
        """Validates that bounds have the correct shape."""
        if self.bounds.shape[1] != 2:
            raise ValueError(f"specified bounds are not of shape (-1, 2), but {self.bounds.shape}")

        if self.bounds.shape[0] != X.shape[1]:
            raise ValueError(f"bounds- and input data dimension mismatch: {self.bounds.shape[0]} != {X.shape[1]}")


class OrderedBound(MatchingFunction):
    """
    A standard interval-based matching function producing multi-dimensional
    hyperrectangular conditions. In effect, a lower (l) and upper bound (u) are
    specified for each dimension. Those bounds always fulfill l <= u
    An example x is matched if l_i <= x_i <= u_i for all dimensions i
    """

    def __init__(self, bounds: np.ndarray = np.array([])):
        self.bounds = bounds

    def __call__(self, X: np.ndarray):
        return np.all((self.bounds[:, 0] <= X) &
                      (X <= self.bounds[:, 1]), axis=1)

    @property
    def volume_(self):
        """Calculates the volume of the interval."""
        diff = self.bounds[:, 1] - self.bounds[:, 0]
        return np.prod(diff)

    def copy(self):
        return OrderedBound(self.bounds.copy())

    def clip(self, bounds: np.ndarray):
        low, high = bounds[None].T
        self.bounds.clip(low, high, out=self.bounds)

    def min_range(self, min_range: float):
        diff = self.bounds[:, 1] - self.bounds[:, 0]
        if min_range > 0:
            invalid_indices = np.argwhere(diff < min_range)
            self.bounds[invalid_indices, 0] -= min_range / 2
            self.bounds[invalid_indices, 1] += min_range / 2


class UnorderedBound(MatchingFunction):
    """
    A standard interval-based matching function producing multi-dimensional
    hyperrectangular conditions. Two bounds (p and q) exist which have
    no explicit ordering but are instead sorted during the matching process
    An example x is matched if q_i <= x_i <= p_i or p_i <= x_i <= q_i for all dimensions i
    """

    def __init__(self, bounds: np.ndarray):
        self.bounds = bounds

    def __call__(self, X: np.ndarray):
        lower = np.min(self.bounds, axis=1)
        upper = np.max(self.bounds, axis=1)
        return np.all((lower <= X) & (X <= upper), axis=1)

    @property
    def volume_(self):
        """Calculates the volume of the interval."""
        diff = self.bounds[:, 1] - self.bounds[:, 0]
        diff = np.maximum(diff, -diff)
        return np.prod(diff)

    def copy(self):
        return UnorderedBound(self.bounds.copy())

    def clip(self, bounds: np.ndarray):
        self.bounds.clip(-1, 1)

    def min_range(self, min_range: float):
        diff = self.bounds[:, 1] - self.bounds[:, 0]
        if min_range > 0:
            invalid_indices = np.argwhere((diff < min_range) & (-diff < min_range))
            # Select indices where p >= q and diff < min_range
            invalid_indices_l = np.argwhere((diff[invalid_indices] > -diff[invalid_indices]))
            # Select indices where p <= q and diff < min_range
            invalid_indices_r = np.argwhere((diff[invalid_indices] <= -diff[invalid_indices]))

            # Increase Range for indices where p >= q
            self.bounds[invalid_indices_l, 0] += min_range / 2
            self.bounds[invalid_indices_l, 1] -= min_range / 2

            # Increase Range for indices where q <= q
            self.bounds[invalid_indices_r, 0] -= min_range / 2
            self.bounds[invalid_indices_r, 1] += min_range / 2


class CenterSpread(MatchingFunction):
    """
    A standard interval-based matching function producing multi-dimensional
    hyperrectangular conditions. In effect, a centre (c) and
    a spread (s) are specified for each dimension.
    An example x is matched iff c_i - s_i <= x_i <= c_i + s_i for all dimensions i
    """

    def __init__(self, bounds: np.ndarray):
        self.bounds = bounds

    def __call__(self, X: np.ndarray):
        return np.all(((self.bounds[:, 0] - self.bounds[:, 1]) <= X) &
                      (X <= (self.bounds[:, 0] + self.bounds[:, 1])), axis=1)

    def calculate_widths(self):
        """Calculates the individual widths"""
        lower = self.bounds[:, 0] - self.bounds[:, 1]
        higher = self.bounds[:, 0] + self.bounds[:, 1]
        lower = lower.clip(-1, 1)
        higher = higher.clip(-1, 1)
        return higher - lower

    @property
    def volume_(self):
        """Calculates the volume of the interval."""
        return np.prod(self.calculate_widths())

    def copy(self):
        return CenterSpread(self.bounds.copy())

    def clip(self, bounds: np.ndarray):
        self.bounds[:, 0] = self.bounds[:, 0].clip(-1, 1)
        self.bounds[:, 1] = self.bounds[:, 1].clip(0, 2)

    def min_range(self, min_range: float):
        diff = self.calculate_widths()
        if min_range > 0:
            invalid_indices = np.argwhere(diff < min_range)
            self.bounds[invalid_indices, 1] += min_range / 2


class MinPercentage(MatchingFunction):
    """
    A standard interval-based matching function producing multi-dimensional
    hyperrectangular conditions. In effect, a lower bound (l) and
    a distance proportion (p) are specified for each dimension.
    An example x is matched if
    l_i <= x_i <= l_i + p_i * (max_i - l_i) for all dimensions i
    """

    def __init__(self, bounds: np.ndarray):
        self.bounds = bounds

    def __call__(self, X: np.ndarray):
        lower = self.bounds[:, 0]
        upper = lower + self.bounds[:, 1] * (1 - lower)
        return np.all((lower <= X) & (X <= upper), axis=1)

    def calculate_widths(self):
        """Calculates the individual widths"""
        lower = self.bounds[:, 0]
        higher = lower + self.bounds[:, 1] * (1 - (-1))
        return higher - lower

    @property
    def volume_(self):
        """Calculates the volume of the interval."""
        return np.prod(self.calculate_widths())

    def copy(self):
        return MinPercentage(self.bounds.copy())

    def clip(self, bounds: np.ndarray):
        self.bounds[:, 0] = self.bounds[:, 0].clip(-1, 1)
        self.bounds[:, 1] = self.bounds[:, 1].clip(0, 1)

    def min_range(self, min_range: float):
        diff = self.calculate_widths()
        if min_range > 0:
            invalid_indices = np.argwhere(diff < min_range)
            # Approximate increasing the width by min_range
            self.bounds[invalid_indices, 0] -= min_range / 2
            self.bounds[invalid_indices, 1] += min_range
