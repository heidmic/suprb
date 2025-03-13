import numpy as np

from suprb.solution.initialization import SolutionInit, RandomInit, Solution
from suprb.utils import flatten
from .crossover import SolutionCrossover, NPoint
from .mutation import SolutionMutation, BitFlips
from .selection import SolutionSelection, Tournament
from ..archive import SolutionArchive, Elitist
from ..base import PopulationBasedSolutionComposition


class GeneticAlgorithm(PopulationBasedSolutionComposition):
    """A simple Genetic Algorithm.

    Parameters
    ----------
    n_iter: int
        Iterations the the metaheuristic will perform.
    population_size: int
        Number of solutions in the population.
    mutation: SolutionMutation
    crossover: SolutionCrossover
    selection: SolutionSelection
    init: SolutionInit
    archive: SolutionArchive
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
    warm_start: bool
        If False, solutions are generated new for every `optimize()` call.
        If True, solutions are used from previous runs.
    n_jobs: int
        The number of threads / processes the optimization uses.
    """

    n_elitists_: int

    def __init__(
        self,
        n_iter: int = 32,
        population_size: int = 32,
        elitist_ratio: float = 0.17,
        mutation: SolutionMutation = BitFlips(),
        crossover: SolutionCrossover = NPoint(n=3),
        selection: SolutionSelection = Tournament(),
        init: SolutionInit = RandomInit(),
        archive: SolutionArchive = Elitist(),
        random_state: int = None,
        n_jobs: int = 1,
        warm_start: bool = True,
        mutation_rate: float = 0.001,
        crossover_rate: float = 0.9,
    ):
        super().__init__(
            n_iter=n_iter,
            population_size=population_size,
            init=init,
            archive=archive,
            random_state=random_state,
            n_jobs=n_jobs,
            warm_start=warm_start,
        )

        self.mutation = mutation
        self.crossover = crossover
        self.selection = selection
        self.elitist_ratio = elitist_ratio
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def adjust_rates(self):
        pass

    def fitness_calculation(self):
        pass

    def crossover_children(self, parent_pairs):
        return list(
            flatten(
                [
                    (
                        self.crossover(A, B, self.crossover_rate, random_state=self.random_state_),
                        self.crossover(B, A, self.crossover_rate, random_state=self.random_state_),
                    )
                    for A, B in parent_pairs
                ]
            )
        )

    def mutate_children(self, children):
        return [self.mutation(child, self.mutation_rate, random_state=self.random_state_) for child in children]

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        assert self.population_size % 2 == 0
        self.fit_population(X, y)

        self.n_elitists_ = int(self.population_size * self.elitist_ratio)

        for _ in range(self.n_iter):
            self.fitness_calculation()

            # Adjust Rates for current fitness
            self.adjust_rates()

            # Eltitism
            elitists = sorted(self.population_, key=lambda i: i.fitness_, reverse=True)[: self.n_elitists_]

            # Selection
            parents = self.selection(
                population=self.population_,
                n=self.population_size,
                random_state=self.random_state_,
            )

            # Note that this expression swallows the last element, if `population_size` is odd
            parent_pairs = map(lambda *x: x, *([iter(parents)] * 2))

            # Crossover
            children = self.crossover_children(parent_pairs)

            # Mutation
            mutated_children = self.mutate_children(children)

            # Replacement
            self.population_ = elitists
            self.population_.extend(mutated_children)

            self.fit_population(X, y)
