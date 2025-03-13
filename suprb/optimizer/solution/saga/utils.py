from __future__ import annotations

import numpy as np


from suprb.optimizer.solution.ga.crossover import NPoint, SolutionCrossover

from suprb.solution.initialization import Solution

import numpy as np
from sklearn.metrics import mean_squared_error

from suprb.rule import Rule
from suprb.solution.base import MixingModel, Solution, SolutionFitness
from suprb.base import BaseComponent
from suprb.utils import RandomState
from suprb.optimizer.solution.archive import SolutionArchive
from suprb.solution.initialization import SolutionInit


class SagaSolution(Solution):
    """Solution that mixes a subpopulation of rules. Extended to have a individual mutationrate, crossoverrate and crossovermethod"""

    def __init__(
        self,
        genome: np.ndarray,
        pool: list[Rule],
        mixing: MixingModel,
        fitness: SolutionFitness,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.001,
        crossover_method: SolutionCrossover = NPoint(n=3),
        age: int = 3,
    ):
        super().__init__(genome, pool, mixing, fitness)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.crossover_method = crossover_method
        self.age = age

    def fit(self, X: np.ndarray, y: np.ndarray) -> SagaSolution:
        pred = self.predict(X, cache=True)
        self.error_ = max(mean_squared_error(y, pred), 1e-4)
        self.input_size_ = self.genome.shape[0]
        self.complexity_ = np.sum(self.genome).item()  # equivalent to np.count_nonzero, but possibly faster
        self.fitness_ = self.fitness(self)
        self.is_fitted_ = True
        return self

    def clone(self, **kwargs) -> SagaSolution:
        args = dict(
            genome=self.genome.copy() if "genome" not in kwargs else None,
            pool=self.pool,
            mixing=self.mixing,
            fitness=self.fitness,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            crossover_method=self.crossover_method,
        )
        solution = SagaSolution(**(args | kwargs))
        if not kwargs:
            attributes = [
                "fitness_",
                "error_",
                "complexity_",
                "is_fitted_",
                "input_size_",
            ]
            solution.__dict__ |= {key: getattr(self, key) for key in attributes}
        return solution


class Uniform(SolutionCrossover):
    """Decide for every bit with uniform probability if the bit in genome A or B is used."""

    def _crossover(self, A: SagaSolution, B: SagaSolution, random_state: RandomState) -> SagaSolution:
        indices = random_state.random(size=len(A.genome)) <= 0.5
        genome = np.empty(A.genome.shape)
        genome[indices] = A.genome[indices]
        genome[~indices] = B.genome[~indices]

        return A.clone(genome=genome)


class SagaElitist(SolutionArchive):

    def __call__(self, new_population: list[SagaSolution]):
        best = max(new_population, key=lambda i: i.fitness_)
        if self.population_:
            if self.population_[0].fitness_ < best.fitness_:
                self.population_.pop(0)
                self.population_.append(best.clone())
        else:
            self.population_.append(best.clone())


def padding_size(solution: SagaSolution) -> int:
    """Calculates the number of bits to add to the genome after the pool was expanded."""

    return len(solution.pool) - solution.genome.shape[0]


def random(n: int, p: float, random_state: RandomState):
    """Returns a random bit string of size `n`, with ones having probability `p`."""

    return (random_state.random(size=n) <= p).astype("bool")


class SagaRandomInit(SolutionInit):
    """Init and extend genomes with random values, with `p` denoting the probability of ones."""

    def __init__(
        self,
        mixing: MixingModel = None,
        fitness: SolutionFitness = None,
        p: float = 0.5,
    ):
        super().__init__(mixing=mixing, fitness=fitness)
        self.p = p

    def __call__(self, pool: list[Rule], random_state: RandomState) -> SagaSolution:
        return SagaSolution(
            genome=random(len(pool), self.p, random_state),
            pool=pool,
            mixing=self.mixing,
            fitness=self.fitness,
        )

    def pad(self, solution: SagaSolution, random_state: RandomState) -> SagaSolution:
        solution.genome = np.concatenate(
            (solution.genome, random(padding_size(solution), self.p, random_state)),
            axis=0,
        )
        return solution
