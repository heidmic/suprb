from abc import ABCMeta, abstractmethod

import numpy as np

from suprb.base import BaseComponent
from suprb.rule import Rule
from . import Solution, MixingModel, SolutionFitness
from .fitness import ComplexityWu
from .mixing_model import ErrorExperienceHeuristic
from ..utils import RandomState


def padding_size(solution: Solution) -> int:
    """Calculates the number of bits to add to the genome after the pool was expanded."""

    return len(solution.pool) - solution.genome.shape[0]


def random(n: int, p: float, random_state: RandomState):
    """Returns a random bit string of size `n`, with ones having probability `p`."""

    return (random_state.random(size=n) <= p).astype("bool")


class SolutionInit(BaseComponent, metaclass=ABCMeta):
    """Generates initial genomes and pads existing genomes."""

    def __init__(self, mixing: MixingModel = None, fitness: SolutionFitness = None):
        self.mixing = mixing
        self.fitness = fitness

        self._validate_components(mixing=ErrorExperienceHeuristic(), fitness=ComplexityWu())

    @abstractmethod
    def __call__(self, pool: list[Rule], random_state: RandomState) -> Solution:
        pass

    @abstractmethod
    def pad(self, solution: Solution, random_state: RandomState) -> Solution:
        pass


class ZeroInit(SolutionInit):
    """Init and extend genomes with zeros."""

    def __call__(self, pool: list[Rule], random_state: RandomState) -> Solution:
        return Solution(
            genome=np.zeros(len(pool), dtype="bool"),
            pool=pool,
            mixing=self.mixing,
            fitness=self.fitness,
        )

    def pad(self, solution: Solution, random_state: RandomState = None) -> Solution:
        solution.genome = np.pad(solution.genome, (0, padding_size(solution)), mode="constant")
        return solution


class RandomInit(SolutionInit):
    """Init and extend genomes with random values, with `p` denoting the probability of ones."""

    def __init__(
        self,
        mixing: MixingModel = None,
        fitness: SolutionFitness = None,
        p: float = 0.5,
    ):
        super().__init__(mixing=mixing, fitness=fitness)
        self.p = p

    def __call__(self, pool: list[Rule], random_state: RandomState) -> Solution:
        return Solution(
            genome=random(len(pool), self.p, random_state),
            pool=pool,
            mixing=self.mixing,
            fitness=self.fitness,
        )

    def pad(self, solution: Solution, random_state: RandomState) -> Solution:
        solution.genome = np.concatenate(
            (solution.genome, random(padding_size(solution), self.p, random_state)),
            axis=0,
        )
        return solution
