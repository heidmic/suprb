import numpy as np

from suprb.utils import RandomState


def sigmoid(x: np.ndarray, scale: float = 10, offset: float = 0.5) -> np.ndarray:
    return 1 / (1 + np.exp(-scale * (x - offset)))


def sigmoid_binarize(x: np.ndarray, random_state: RandomState, **kwargs) -> np.ndarray:
    return (random_state.random(size=x.shape[0]) <= sigmoid(x, **kwargs)).astype(np.bool)
