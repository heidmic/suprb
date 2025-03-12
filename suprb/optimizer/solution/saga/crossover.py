import numpy as np
from abc import ABCMeta, abstractmethod
from suprb.base import BaseComponent
from .solution_extension import SagaSolution, NPoint, Uniform
from suprb.utils import RandomState


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


class SolutionCrossover(BaseComponent, metaclass=ABCMeta):

    def __init__(self, crossover_rate: float = 0.9):
        self.crossover_rate = crossover_rate

    def __call__(self, A: SagaSolution, B: SagaSolution, random_state: RandomState) -> SagaSolution:
        if random_state.random() < self.crossover_rate:
            return self._crossover(A=A, B=B, random_state=random_state)
        else:
            # Just return the primary parent
            return A

    @abstractmethod
    def _crossover(self, A: SagaSolution, B: SagaSolution, random_state: RandomState) -> SagaSolution:
        pass


class NPoint(SolutionCrossover):
    """Cut the genome at N points and alternate the pieces from solution A and B."""

    def __init__(self, crossover_rate: float = 0.9, n: int = 2):
        super().__init__(crossover_rate=crossover_rate)
        self.n = n

    @staticmethod
    def _single_point(A: SagaSolution, B: SagaSolution, index: int) -> SagaSolution:
        return A.clone(genome=np.append(A.genome[:index], B.genome[index:]))

    def _crossover(self, A: SagaSolution, B: SagaSolution, random_state: RandomState) -> SagaSolution:
        indices = random_state.choice(np.arange(len(A.genome)), size=min(self.n, len(A.genome)), replace=False)
        for index in indices:
            A = self._single_point(A, B, index)
        return A


class Uniform(SolutionCrossover):
    """Decide for every bit with uniform probability if the bit in genome A or B is used."""

    def _crossover(self, A: SagaSolution, B: SagaSolution, random_state: RandomState) -> SagaSolution:
        indices = random_state.random(size=len(A.genome)) <= 0.5
        genome = np.empty(A.genome.shape)
        genome[indices] = A.genome[indices]
        genome[~indices] = B.genome[~indices]

        return A.clone(genome=genome)
