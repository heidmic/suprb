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
    An example x is matched iff l_i <= x_i <= u_i for all dimensions i
    """

    def __init__(self, bounds: np.ndarray | None = None):
        self.bounds = np.array([]) if bounds is None else bounds

    def __call__(self, X: np.ndarray):
        return np.all((self.bounds[:, 0] <= X) & (X <= self.bounds[:, 1]), axis=1)

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
    An example x is matched iff q_i <= x_i <= p_i for all dimensions i
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
        return np.all(
            ((self.bounds[:, 0] - self.bounds[:, 1]) <= X) & (X <= (self.bounds[:, 0] + self.bounds[:, 1])),
            axis=1,
        )

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
    An example x is matched iff
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

class GABIL(MatchingFunction):
    """A bitstring condition over a single categorical attribute.

    Real-valued conditions in SupRB use an interval per dimension. That is
    meaningless for categorical data, since categories have no order: there
    is no sense in which "Berlin" lies between "Lahore" and "Tokyo".

    GABIL (De Jong & Spears, 1991) instead stores one bit per category level.
    A set bit means the rule matches that level:

        levels        [Lahore, Karachi, Berlin, Tokyo]
        bits          [  1   ,    0   ,   1   ,   0  ]
        reads as      "city is Lahore or Berlin"

    Two states are special:
        all bits set    -> matches everything (the "don't care" state)
        no bits set     -> matches nothing, which is an invalid rule
    """

    def __init__(self, bits: np.ndarray = None):
        # Stored as a plain boolean array of length k (k = number of levels).
        # The empty default mirrors OrderedBound, which starts with empty
        # bounds and has them filled in by rule initialisation.
        self.bits = np.array([], dtype=bool) if bits is None else bits

    def __call__(self, X: np.ndarray):
        """Return a boolean array: True where the rule matches the sample.

        A bitstring condition governs exactly one categorical attribute, so
        a two-dimensional X is only meaningful when it carries a single
        column. Anything wider needs a composite matching function that
        holds one condition per attribute, which does not exist yet; the
        restriction is therefore reported rather than silently flattened.

        Codes are validated against the bitstring length for the same
        reason: an out-of-range code is a malformed encoding, and clipping
        it would quietly assign the sample to the wrong category.
        """
        X = np.asarray(X)

        if X.ndim == 2 and X.shape[1] != 1:
            raise ValueError(
                f"GABIL matches a single categorical attribute, but X has "
                f"{X.shape[1]} columns. A composite matching function is "
                f"required for mixed-type inputs."
            )

        codes = X.astype(int).reshape(-1)

        if codes.size and (codes.min() < 0 or codes.max() >= self.bits.size):
            raise ValueError(
                f"Category codes must lie in [0, {self.bits.size}), but the "
                f"observed range is [{codes.min()}, {codes.max()}]."
            )

        return self.bits[codes]

    @property
    def volume_(self):
        """How general the rule is, as a fraction of the category space.

        This is the categorical counterpart of interval width. A rule
        matching 2 of 4 levels has volume 0.5, exactly as an interval
        covering half its dimension would.
        """
        return float(self.bits.mean()) if self.bits.size else 0.0

    def copy(self):
        """Return a deep copy.

        The bits array must be copied, not shared. Mutation modifies bits in
        place, so a shared array would let a mutated child silently corrupt
        its parent.
        """
        return GABIL(bits=self.bits.copy())

    def clip(self, bounds: np.ndarray):
        """Repair the rule if it has become invalid.

        Bits are already constrained to {0, 1}, so there is no range to clip
        into. What can go wrong is the all-zero state, which matches nothing.
        The rule is repaired by setting its first bit, keeping the operation
        deterministic so that seeding stays reproducible.
        """
        if self.bits.size and not self.bits.any():
            self.bits[0] = True

    def min_range(self, min_range: float):
        """No-op.

        This exists to stop interval rules from collapsing to near-zero
        width. A bitstring has no continuous extent to widen, and its
        smallest valid state (one bit set) is already meaningful, so there
        is nothing to enforce here.
        """
        pass