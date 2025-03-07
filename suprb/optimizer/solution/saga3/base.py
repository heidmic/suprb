import numpy as np

from suprb.optimizer.solution.saga3.archive import SagaElitist

from .initialization import SagaRandomInit
from suprb.solution.initialization import SolutionInit
from suprb.utils import flatten
from .crossover import SagaCrossover, SolutionCrossover
from .mutation import SolutionMutation, BitFlips
from .selection import SolutionSelection, Tournament, Ageing
from .solution_extension import SagaSolution, NPoint
from ..archive import Elitist, SolutionArchive
from ..base import PopulationBasedSolutionComposition


class SelfAdaptingGeneticAlgorithm(PopulationBasedSolutionComposition):
    """A simple self adapting Genetic Algorithm, implemented acording to 10.1023/A:1022521428870 .

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
    population_: list[SagaSolution]

    def __init__(
        self,
        n_iter: int = 32,
        population_size: int = 32,
        elitist_ratio: float = 0.17,
        mutation: SolutionMutation = BitFlips(),
        crossover: SagaCrossover = SagaCrossover(parameter_mutation_rate=0.05),
        selection: SolutionSelection = Tournament(),
        init: SolutionInit = SagaRandomInit(),
        archive: SolutionArchive = SagaElitist(),
        random_state: int = None,
        n_jobs: int = 1,
        warm_start: bool = True,
        parameter_mutation_rate=0.05,
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
        self.parameter_mutation_rate = parameter_mutation_rate

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        assert self.population_size % 2 == 0
        self.fit_population(X, y)

        self.n_elitists_ = int(self.population_size * self.elitist_ratio)

        for _ in range(self.n_iter):
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
            children = list(
                flatten(
                    [
                        (
                            self.crossover(A, B, self.parameter_mutation_rate, random_state=self.random_state_),
                            self.crossover(B, A, self.parameter_mutation_rate, random_state=self.random_state_),
                        )
                        for A, B in parent_pairs
                    ]
                )
            )

            # Mutation
            mutated_children = [self.mutation(child, random_state=self.random_state_) for child in children]

            # Replacement
            self.population_ = elitists
            self.population_.extend(mutated_children)

            self.fit_population(X, y)


class SasGeneticAlgorithm(PopulationBasedSolutionComposition):
    """A simple self adapting Genetic Algorithm, implemented acording to 10.1007/978-3-642-21219-2_16 .

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
    population_: list[SagaSolution]

    def __init__(
        self,
        n_iter: int = 32,
        initial_population_size: int = 100,
        mutation: SolutionMutation = BitFlips(),
        crossover: SolutionCrossover = NPoint(n=3),
        selection: SolutionSelection = Ageing(),
        init: SolutionInit = SagaRandomInit(),
        archive: SolutionArchive = SagaElitist(),
        random_state: int = None,
        n_jobs: int = 1,
        warm_start: bool = True,
        mutation_rate: float = 0.001,
        crossover_rate: float = 0.9,
    ):
        super().__init__(
            n_iter=n_iter,
            population_size=initial_population_size,
            init=init,
            archive=archive,
            random_state=random_state,
            n_jobs=n_jobs,
            warm_start=warm_start,
        )

        self.initial_population_size = initial_population_size
        self.mutation = mutation
        self.crossover = crossover
        self.selection = selection
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        self.fit_population(X, y)

        for _ in range(self.n_iter):
            # Selection
            parents = self.selection(
                population=self.population_,
                initial_population_size=self.initial_population_size,
                random_state=self.random_state_,
            )

            # Correct value for population_size
            self.population_size = len(parents)

            # Note that this expression swallows the last element, if `population_size` is odd
            parent_pairs = map(lambda *x: x, *([iter(parents)] * 2))

            # Crossover
            children = list(
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

            # Mutation
            mutated_children = [
                self.mutation(child, self.mutation_rate, random_state=self.random_state_) for child in children
            ]

            # Replacement
            self.population_ = parents
            self.population_.extend(mutated_children)

            self.fit_population(X, y)
