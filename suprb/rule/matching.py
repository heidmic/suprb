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

    def __init__(self, bits: list = None):
        # One bitstring per categorical attribute, mirroring how
        # OrderedBound stores one [lower, upper] pair per dimension.
        # Bitstrings may differ in length, since attributes can have
        # different numbers of levels. The empty default is filled in by
        # rule initialisation, as with the interval representations.
        self.bits = [] if bits is None else bits

    def __call__(self, X: np.ndarray):
        """Return a boolean array: True where the rule matches the sample.

        A sample matches only if every attribute matches, which is the
        categorical counterpart of a point lying inside every interval of
        a hyperrectangle.

        The number of columns must equal the number of bitstrings. A
        mismatch means the input does not correspond to the data the rule
        was built on, so it is reported rather than padded or truncated:
        either would silently score the sample against the wrong
        attributes.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != len(self.bits):
            raise ValueError(f"GABIL holds {len(self.bits)} bitstrings but X has " f"{X.shape[1]} columns.")

        if not self.bits:
            return np.ones(X.shape[0], dtype=bool)

        codes = X.astype(int)
        matched = np.ones(X.shape[0], dtype=bool)

        for column, bitstring in enumerate(self.bits):
            column_codes = codes[:, column]

            if column_codes.min() < 0 or column_codes.max() >= bitstring.size:
                raise ValueError(
                    f"Category codes in column {column} must lie in "
                    f"[0, {bitstring.size}), but the observed range is "
                    f"[{column_codes.min()}, {column_codes.max()}]."
                )

            matched &= bitstring[column_codes]

        return matched

    @property
    def volume_(self):
        """How general the rule is, as a fraction of the category space.

        This is the categorical counterpart of interval width. A rule
        matching 2 of 4 levels covers half that attribute, exactly as an
        interval spanning half its range does.

        Attributes are combined by multiplication, mirroring how the
        volume of a hyperrectangle is the product of its side lengths.
        """
        if not self.bits:
            return 0.0

        volume = 1.0
        for bitstring in self.bits:
            if bitstring.size == 0:
                return 0.0
            volume *= float(bitstring.mean())

        return volume

    def copy(self):
        """Return a deep copy.

        Every bitstring must be copied, not just the list holding them.
        Mutation modifies bits in place, so a shallow list copy would
        share the underlying arrays and let a mutated child silently
        corrupt its parent.
        """
        return GABIL(bits=[bitstring.copy() for bitstring in self.bits])

    def clip(self, bounds: np.ndarray):
        """No-op.

        The real-valued version keeps a bound from drifting outside the
        feature range, for instance to 1.000004 on a unit interval. Bits
        are drawn from {0, 1} by construction, so there is no range to
        clip into.
        """
        pass

    def min_range(self, min_range: float):
        """Repair rules that have become too specific to be valid.

        For intervals this widens a bound that has collapsed to near-zero
        width. The categorical counterpart of a degenerate rule is an
        all-zero bitstring, which matches nothing, so each attribute is
        repaired independently.

        The first bit is set rather than a random one, since this method
        receives no random_state and a non-deterministic repair would
        break seeding.
        """
        for bitstring in self.bits:
            if bitstring.size and not bitstring.any():
                bitstring[0] = True


class MixedTypes(MatchingFunction):
    """A condition over a mix of real-valued and categorical attributes.

    Neither OrderedBound nor GABIL can describe a dataset that contains
    both kinds of attribute: intervals are meaningless for categories,
    and bitstrings are meaningless for continuous values. Rather than
    reimplementing both, this composes them and routes each attribute to
    whichever representation suits it:

        mask       [False, True, False]     True marks a categorical column
        reals      OrderedBound over columns 0 and 2
        categoricals  GABIL over column 1

    A sample matches only if both halves match, so this behaves like a
    hyperrectangle in the real dimensions and a set membership test in the
    categorical ones.

    `mask` marks the categorical columns rather than the real ones because
    intervals are the default representation throughout SupRB, so the mask
    identifies the exception.
    """

    def __init__(
        self,
        reals: MatchingFunction = None,
        categoricals: MatchingFunction = None,
        mask: np.ndarray = None,
    ):
        # Stored unchanged, since sklearn's clone reconstructs a component
        # from exactly the values its constructor was given.
        self.reals = reals
        self.categoricals = categoricals
        self.mask = np.array([], dtype=bool) if mask is None else mask

    def __call__(self, X: np.ndarray):
        """Return a boolean array: True where the rule matches the sample.

        Both halves must match. This is the same rule that already applies
        within each representation, where every dimension of a
        hyperrectangle and every attribute of a bitstring has to hold.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.mask.size:
            raise ValueError(
                f"MixedTypes was built for {self.mask.size} attributes but " f"X has {X.shape[1]} columns."
            )

        mask = np.asarray(self.mask, dtype=bool)
        return self.reals(X[:, ~mask]) & self.categoricals(X[:, mask])

    @property
    def volume_(self):
        """The share of the input space the rule covers.

        The two halves multiply, since together they describe one region:
        the real half contributes the volume of a hyperrectangle and the
        categorical half the fraction of category combinations matched.

        Note that these are not in the same units. The interval half is an
        absolute width while the categorical half is normalised, so the
        product is not directly comparable to input_space_volume in
        VolumeRuleFitness. See issue #213.
        """
        return self.reals.volume_ * self.categoricals.volume_

    def copy(self):
        """Return a deep copy.

        Both halves must be copied. Mutation modifies them in place, so
        sharing either one would let a mutated child corrupt its parent.
        The mask is copied as well, even though nothing mutates it, so that
        a copy is independent in every respect.
        """
        return MixedTypes(
            reals=self.reals.copy(),
            categoricals=self.categoricals.copy(),
            mask=self.mask.copy(),
        )

    def clip(self, bounds: np.ndarray):
        """Keep the interval half inside the feature ranges.

        `bounds` covers every attribute, so it is sliced down to the real
        columns before being passed on. Handing over the full array would
        clip each interval against the wrong feature's range.

        The categorical half needs no clipping, since bits are drawn from
        {0, 1} by construction.
        """
        mask = np.asarray(self.mask, dtype=bool)
        self.reals.clip(bounds[~mask])
        self.categoricals.clip(None)

    def min_range(self, min_range: float):
        """Repair rules that have become too specific to be valid.

        Both halves are repaired, since matching ANDs them together: a
        collapsed interval and an all-zero bitstring each make the whole
        rule match nothing.
        """
        self.reals.min_range(min_range)
        self.categoricals.min_range(min_range)
