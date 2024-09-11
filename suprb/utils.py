import collections.abc
import numbers
from typing import Iterator, Union

import numpy as np

RandomState = Union[np.random.RandomState, np.random.Generator]


def check_random_state(seed) -> RandomState:
    """Turn seed into a np.random.Generator or np.random.RandomState instance.

    Note that sklearn currently doesn't support np.random.Generator in its sklearn.utils.check_random_state function.
    See https://github.com/scikit-learn/scikit-learn/issues/16988 for the current status.
    If they actually add support in the future, this function may become obsolete.

    Parameters
    ----------
    seed : None, int or instance of RandomState, Generator
        If seed is None, return a Generator seeded by numpy.
        If seed is an int or SeedSequence, return a new Generator instance seeded with seed.
        If seed is a RandomState instance, return a new Generator instance seeded with its SeedSequence.
        If seed is already a Generator instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or isinstance(seed, numbers.Integral) or isinstance(seed, np.random.SeedSequence):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.RandomState):
        return np.random.default_rng(seed.bit_generator._seed_seq)
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.Generator'
                     ' instance' % seed)


def spawn_random_states(random_state: RandomState, n: int) -> Iterator[RandomState]:
    children = random_state.bit_generator._seed_seq.spawn(n)
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
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el
