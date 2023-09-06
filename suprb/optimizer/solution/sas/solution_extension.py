from __future__ import annotations

import itertools
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error
from suprb.base import BaseComponent
from suprb.optimizer.solution.archive import Elitist

from suprb.rule import Rule
from suprb.solution.base import MixingModel, Solution, SolutionFitness
from suprb.solution.initialization import RandomInit
from suprb.utils import RandomState


def padding_size(solution: Solution) -> int:
    """Calculates the number of bits to add to the genome after the pool was expanded."""

    return len(solution.pool) - solution.genome.shape[0]


def random(n: int, p: float, random_state: RandomState):
    """Returns a random bit string of size `n`, with ones having probability `p`."""

    return (random_state.random(size=n) <= p).astype('bool')
    

class SasSolution(Solution):
    """Solution that mixes a subpopulation of rules. Extended to have a individual mutationrate, crossoverrate and crossovermethod"""

    def __init__(self, genome: np.ndarray, 
                 pool: list[Rule], 
                 mixing: MixingModel, 
                 fitness: SolutionFitness,
                 age: int = 3):
        super().__init__(genome, pool, mixing, fitness)
        self.age = age


    def clone(self, **kwargs) -> SasSolution:
        args = dict(
            genome=self.genome.copy() if 'genome' not in kwargs else None,
            pool=self.pool,
            mixing=self.mixing,
            fitness=self.fitness,
        )
        solution = SasSolution(**(args | kwargs))
        if not kwargs:
            attributes = ['fitness_', 'error_', 'complexity_', 'is_fitted_', 'input_size_']
            solution.__dict__ |= {key: getattr(self, key) for key in attributes}
        return solution


class SasRandominit(RandomInit):
    """Init and extend genomes with random values, with `p` denoting the probability of ones."""

    def __call__(self, pool: list[Rule], random_state: RandomState) -> SasSolution:
        return SasSolution(genome=random(len(pool), self.p, random_state), pool=pool, mixing=self.mixing,
                        fitness=self.fitness)

    def pad(self, solution: SasSolution, random_state: RandomState) -> SasSolution:
        solution.genome = np.concatenate((solution.genome, random(padding_size(solution), self.p, random_state)),
                                         axis=0)
        return solution
    

class SasElitist(Elitist):
    
    population_: list[SasSolution]

    def __call__(self, new_population: list[SasSolution]):
        best = max(new_population, key=lambda i: i.fitness_)
        if self.population_:
            if self.population_[0].fitness_ < best.fitness_:
                self.population_.pop(0)
                self.population_.append(best.clone())
        else:
            self.population_.append(best.clone())