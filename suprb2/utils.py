import collections
import numbers
from typing import Iterator, Union

import numpy as np


def check_random_state(seed) -> Union[np.random.Generator, np.random.RandomState]:
    """Turn seed into a np.random.Generator or np.random.RandomState instance.

    Note that sklearn currently doesn't support np.random.Generator in its sklearn.utils.check_random_state function.
    See https://github.com/scikit-learn/scikit-learn/issues/16988 for the current status.
    If they actually add support in the future, this function may become obsolete.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new Generator instance seeded with seed.
        If seed is already a RandomState or Generator instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.RandomState) or isinstance(seed, np.random.Generator):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def spawn_random_states(ss: np.random.SeedSequence, n: int) -> Iterator[np.random.RandomState]:
    children = ss.spawn(n)
    for child in children:
        yield np.random.default_rng(child)


def estimate_bounds(X) -> np.ndarray:
    return np.stack((np.min(X, axis=0), np.max(X, axis=0)), axis=0).T


def flatten(iterable):
    """
    Flattens an iterable that itself contains lists or single elements.
    Note that implementations like `itertools.chain` only flatten nested lists, not irregular nested lists.
     """
    for el in iterable:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el
