from suprb.base import BaseComponent
from .solution_extension import SagaSolution, NPoint, Uniform
from suprb.utils import RandomState


class SagaCrossover(BaseComponent):
    """Performs crossover and mutation of Parameters, then calls resulting crossover function"""

    def __init__(self, parameter_mutation_rate: float):
        self.parameter_mutation_rate = parameter_mutation_rate

    def __call__(self, A: SagaSolution, B: SagaSolution, random_state: RandomState) -> SagaSolution:
        # Crossover of parent parameters
        new_crossover_rate = random_state.choice([A.crossover_rate, B.crossover_rate])
        new_mutation_rate = random_state.choice([A.mutation_rate, B.mutation_rate])
        new_crossover_method = random_state.choice(
            [A.crossover_method, B.crossover_method]
        )

        # Mutation of parameters
        if random_state.random() < self.parameter_mutation_rate:
            new_crossover_rate = min(
                max(new_crossover_rate + random_state.normal(), 0.0), 1.0
            )
            new_mutation_rate = min(
                max(new_mutation_rate + random_state.normal(), 0.0), 1.0
            )
            new_crossover_method = random_state.choice([NPoint(n=3), Uniform()])

        # Crossover of genome
        new_solution: SagaSolution = new_crossover_method(A, B, new_crossover_rate, random_state=random_state)
        new_solution.crossover_rate = new_crossover_rate
        new_solution.mutation_rate = new_mutation_rate
        new_solution.crossover_method = new_crossover_method
        return new_solution
