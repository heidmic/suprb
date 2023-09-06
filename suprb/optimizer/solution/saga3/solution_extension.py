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
    

class SolutionCrossover(BaseComponent, metaclass=ABCMeta):

    def __init__(self):
        pass

    def __call__(self, A: SagaSolution, B: SagaSolution, crossover_rate: float, random_state: RandomState) -> SagaSolution:
        if random_state.random() < crossover_rate:
            return self._crossover(A=A, B=B, random_state=random_state)
        else:
            # Just return the primary parent
            return A

    @abstractmethod
    def _crossover(self, A: SagaSolution, B: SagaSolution, random_state: RandomState) -> SagaSolution:
        pass


class NPoint(SolutionCrossover):
    """Cut the genome at N points and alternate the pieces from solution A and B."""

    def __init__(self, n: int = 2):
        super().__init__()
        self.n = n

    @staticmethod
    def _single_point(A: SagaSolution, B: SagaSolution, index: int) -> SagaSolution:
        return A.clone(genome=np.append(A.genome[:index], B.genome[index:]))

    def _crossover(self, A: SagaSolution, B: SagaSolution, random_state: RandomState) -> SagaSolution:
        indices = random_state.choice(np.arange(len(A.genome)), size=min(self.n, len(A.genome)), replace=False)
        for index in indices:
            A = self._single_point(A, B, index)
            B = self._single_point(B, A, index)
        return A


class Uniform(SolutionCrossover):
    """Decide for every bit with uniform probability if the bit in genome A or B is used."""

    def _crossover(self, A: SagaSolution, B: SagaSolution, random_state: RandomState) -> SagaSolution:
        indices = random_state.random(size=len(A.genome)) <= 0.5
        genome = np.empty(A.genome.shape)
        genome[indices] = A.genome[indices]
        genome[~indices] = B.genome[~indices]

        return A.clone(genome=genome)


class SagaSolution(Solution):
    """Solution that mixes a subpopulation of rules. Extended to have a individual mutationrate, crossoverrate and crossovermethod"""

    def __init__(self, genome: np.ndarray, 
                 pool: list[Rule], 
                 mixing: MixingModel, 
                 fitness: SolutionFitness, 
                 crossover_rate: float = 0.9, 
                 mutation_rate: float = 0.001, 
                 crossover_method: SolutionCrossover = NPoint(n=3)):
        super().__init__(genome, pool, mixing, fitness)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.crossover_method = crossover_method


    def clone(self, **kwargs) -> SagaSolution:
        args = dict(
            genome=self.genome.copy() if 'genome' not in kwargs else None,
            pool=self.pool,
            mixing=self.mixing,
            fitness=self.fitness,
            crossover_rate = self.crossover_rate,
            mutation_rate = self.mutation_rate,
            crossover_method = self.crossover_method
        )
        solution = SagaSolution(**(args | kwargs))
        if not kwargs:
            attributes = ['fitness_', 'error_', 'complexity_', 'is_fitted_', 'input_size_']
            solution.__dict__ |= {key: getattr(self, key) for key in attributes}
        return solution


class SagaRandominit(RandomInit):
    """Init and extend genomes with random values, with `p` denoting the probability of ones."""

    def __call__(self, pool: list[Rule], random_state: RandomState) -> SagaSolution:
        return SagaSolution(genome=random(len(pool), self.p, random_state), pool=pool, mixing=self.mixing,
                        fitness=self.fitness)

    def pad(self, solution: SagaSolution, random_state: RandomState) -> SagaSolution:
        solution.genome = np.concatenate((solution.genome, random(padding_size(solution), self.p, random_state)),
                                         axis=0)
        return solution
    

class SagaElitist(Elitist):
    
    population_: list[SagaSolution]

    def __call__(self, new_population: list[SagaSolution]):
        best = max(new_population, key=lambda i: i.fitness_)
        if self.population_:
            if self.population_[0].fitness_ < best.fitness_:
                self.population_.pop(0)
                self.population_.append(best.clone())
        else:
            self.population_.append(best.clone())