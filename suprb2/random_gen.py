import numpy as np


class Random:
    """
    We use Alex Martelli's Borg pattern to share our random generator properly
    in a kind of singleton way.

    See https://code.activestate.com/recipes/66531-singleton-we-dont-need-no-stinkin-singleton-the-bo/ .
    """
    __shared_state = {"random": np.random.default_rng(0), "_seed": 0}

    def __init__(self):
        self.__dict__ = self.__shared_state

    def seed(self, seed):
        self.random = np.random.default_rng(seed)
        self._seed = seed
