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


class SagaCrossover(BaseComponent):
    """Performs crossover and mutation of Parameters, then calls resulting crossover function"""

    def __init__(self, parameter_mutation_rate: float):
        self.parameter_mutation_rate = parameter_mutation_rate

    def __call__(self, A: SagaSolution, B: SagaSolution, crossover_rate, random_state: RandomState) -> SagaSolution:
        # Crossover of parent parameters
        new_crossover_rate = random_state.choice([A.crossover_rate, B.crossover_rate])
        new_mutation_rate = random_state.choice([A.mutation_rate, B.mutation_rate])
        new_crossover_method = random_state.choice([A.crossover_method, B.crossover_method])

        # Mutation of parameters
        if random_state.random() < self.parameter_mutation_rate:
            new_crossover_rate = min(max(new_crossover_rate + random_state.normal(), 0.0), 1.0)
            new_mutation_rate = min(max(new_mutation_rate + random_state.normal(), 0.0), 1.0)
            new_crossover_method = random_state.choice([NPoint(n=3), Uniform()])

        # Crossover of genome
        try:
            new_solution: SagaSolution = new_crossover_method(A, B, new_crossover_rate, random_state=random_state)
        except TypeError:
            new_solution: SagaSolution = new_crossover_method(A, B, random_state=random_state)

        new_solution.crossover_rate = new_crossover_rate
        new_solution.mutation_rate = new_mutation_rate
        new_solution.crossover_method = new_crossover_method
        return new_solution


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
