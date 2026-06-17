import numpy as np
import scipy.stats as stats

from suprb import Solution
from suprb.solution.fitness import NormalizedMOSolutionFitness
from suprb.solution.initialization import SolutionInit, RandomInit
from suprb.utils import flatten
from .sorting import fast_non_dominated_sort

from ..base import MOSolutionComposition
from .mutation import SolutionMutation, BitFlips
from .selection import SolutionSelection, BinaryTournament
from .crossover import SolutionCrossover, NPoint
from .archive import EnvironmentalArchive
from ..sampler import SolutionSampler, BetaSolutionSampler
from .internal_fitness import calculate_raw_internal_fitness


class StrengthParetoEvolutionaryAlgorithm2(MOSolutionComposition):
    """Stringth Pareto Evolutionary Algorithm 2.

    Implemented as described in 10.3929/ethz-a-004284029

    Parameters
    ----------
    n_iter: int
        Iterations the metaheuristic will perform.
    population_size: int
        Number of solutions in the population.
    mutation: SolutionMutation
    crossover: SolutionCrossover
    selection: SolutionSelection
    init: SolutionInit
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
    warm_start: bool
        If False, solutions are generated new for every `optimize()` call.
        If True, solutions are used from previous runs.
    n_jobs: int
        The number of threads / processes the optimization uses.
    """

    def __init__(
        self,
        n_iter: int = 32,
        population_size: int = 32,
        archive_size: int = 32,
        mutation: SolutionMutation = BitFlips(),
        crossover: SolutionCrossover = NPoint(n=3),
        selection: SolutionSelection = BinaryTournament(),
        sampler: SolutionSampler = BetaSolutionSampler(1.5, 1.5),
        mutation_rate: float = 0.025,
        crossover_rate: float = 0.75,
        kth_nearest: int = -1,
        init: SolutionInit = RandomInit(fitness=NormalizedMOSolutionFitness()),
        random_state: int = None,
        n_jobs: int = 1,
        warm_start: bool = True,
        early_stopping_patience: int = -1,
        early_stopping_delta: float = 0,
    ):
        super().__init__(
            n_iter=n_iter,
            population_size=population_size,
            init=init,
            archive=EnvironmentalArchive(
                archive_size, kth_nearest if kth_nearest != -1 else int((population_size + archive_size) ** 0.5)
            ),
            sampler=sampler,
            random_state=random_state,
            n_jobs=n_jobs,
            warm_start=warm_start,
            early_stopping_patience=early_stopping_patience,
            early_stopping_delta=early_stopping_delta,
        )
        self.mutation = mutation
        self.crossover = crossover
        self.selection = selection

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.kth_nearest = kth_nearest
        self.archive_size = archive_size

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        self.fit_population(X, y)
        for _ in range(self.n_iter):
            self.archive(self.population_)
            internal_fitness_values = np.array([solution.internal_fitness_ for solution in self.archive.population_])
            parents = self.selection(
                population=self.archive.population_,
                n=self.population_size,
                random_state=self.random_state_,
                internal_fitness=internal_fitness_values,
            )

            # Note that this expression swallows the last element, if `population_size` is odd
            parent_pairs = map(lambda *x: x, *([iter(parents)] * 2))

            # Crossover
            children = list(
                flatten(
                    [
                        (
                            self.crossover(A, B, random_state=self.random_state_),
                            self.crossover(B, A, random_state=self.random_state_),
                        )
                        for A, B in parent_pairs
                    ]
                )
            )
            # If `population_size` is odd, we add the solution not selected for reproduction directly
            if self.population_size % 2 != 0:
                children.append(parents[-1])

            # Mutation
            mutated_children = [self.mutation(child, random_state=self.random_state_).fit(X, y) for child in children]
            self.population_ = mutated_children

            if self.check_early_stopping():
                break

    def pareto_front(self) -> list[Solution]:
        if not hasattr(self, "population_") or not self.population_:
            return []
        fitness_values = np.array([solution.fitness_ for solution in self.archive.population_])
        pareto_ranks = fast_non_dominated_sort(fitness_values)
        pareto_front = np.array(self.archive.population_)[pareto_ranks == 0]
        return sorted(pareto_front, key=lambda x: x.fitness_[0], reverse=True)
