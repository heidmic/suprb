import numbers
from typing import Iterator, Union

import numpy as np


def check_random_state(seed) -> Union[np.random.Generator, np.random.RandomState]:
    """Turn seed into a np.random.RandomState instance

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
