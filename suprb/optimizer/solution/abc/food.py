from abc import ABCMeta

import numpy as np
from suprb import Solution
from suprb.base import BaseComponent
from suprb.utils import RandomState

from suprb.optimizer.solution.utils import sigmoid_binarize


class FoodSource:

    def __init__(self, solution: Solution):
        self.solution = solution
        self.trials = 0

    def __repr__(self):
        return f"<{self.trials}, {self.solution}>"


class FoodSourceUpdate(BaseComponent, metaclass=ABCMeta):

    def __call__(self, own: FoodSource, other: FoodSource, random_state: RandomState) -> Solution:
        pass


class Sigmoid(FoodSourceUpdate):
    """Perform the traditional food source update and binarize using sigmoid."""

    def __call__(self, own: FoodSource, other: FoodSource, random_state: RandomState) -> Solution:
        own_genome = own.solution.genome.astype(np.float)
        other_genome = other.solution.genome.astype(np.float)
        rand = random_state.uniform(-1, 1, size=own_genome.shape[0])

        new = own_genome + rand * (own_genome - other_genome)
        new_genome = sigmoid_binarize(new, random_state=random_state)

        return own.solution.clone(genome=new_genome)


class Bitwise(FoodSourceUpdate):
    """
    Perform the food source update using bitwise operations.

    Taken from https://doi.org/10/f6npmn.
    """

    def __call__(self, own: FoodSource, other: FoodSource, random_state: RandomState) -> Solution:
        own_genome = own.solution.genome
        other_genome = other.solution.genome
        rand = random_state.integers(0, 2, size=own_genome.shape[0])

        new_genome = own_genome ^ (rand & (own_genome | other_genome))

        return own.solution.clone(genome=new_genome)


class DimensionFlips(FoodSourceUpdate):
    """
    Perform the food source update as partial crossover.

    Taken from https://doi.org/10/gnxb6q.

    """

    def __init__(self, flip_rate: float = 0.38):
        self.flip_rate = flip_rate

    def __call__(self, own: FoodSource, other: FoodSource, random_state: RandomState) -> Solution:
        own_genome = own.solution.genome
        other_genome = other.solution.genome
        n = own_genome.shape[0]

        n_flips = np.ceil(n * self.flip_rate).astype(np.int)

        new_genome = own_genome.copy()
        flip_dims = random_state.choice(np.arange(n, dtype=np.int), size=n_flips, replace=False)
        new_genome[flip_dims] = other_genome[flip_dims]

        return own.solution.clone(genome=new_genome)
