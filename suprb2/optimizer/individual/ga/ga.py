import numpy as np
from joblib import Parallel

from suprb2.individual.initialization import IndividualInit, RandomInit
from suprb2.utils import flatten
from .crossover import IndividualCrossover, NPoint
from .mutation import IndividualMutation, BitFlips
from .selection import IndividualSelection, RouletteWheel
from ..archive import IndividualArchive, Elitist
from ..base import PopulationBasedIndividualOptimizer


class GeneticAlgorithm(PopulationBasedIndividualOptimizer):
    """ A simple Genetic Algorithm.

    Parameters
    ----------
    n_iter: int
        Iterations the the metaheuristic will perform.
    population_size: int
        Number of individuals in the population.
    mutation: IndividualMutation
    crossover: IndividualCrossover
    selection: IndividualSelection
    init: IndividualInit
    archive: IndividualArchive
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
    warm_start: bool
        If False, individuals are generated new for every `optimize()` call.
        If True, individuals are used from previous runs.
    n_jobs: int
        The number of threads / processes the optimization uses.
    """

    n_elitists_: int

    def __init__(self,
                 n_iter: int = 128,
                 population_size: int = 128,
                 elitist_ratio: float = 0.1,
                 mutation: IndividualMutation = BitFlips(),
                 crossover: IndividualCrossover = NPoint(),
                 selection: IndividualSelection = RouletteWheel(),
                 init: IndividualInit = RandomInit(),
                 archive: IndividualArchive = Elitist(),
                 random_state: int = None,
                 n_jobs: int = 1,
                 warm_start: bool = True,
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

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        self.fit_population(X, y)

        self.n_elitists_ = int(self.population_size * self.elitist_ratio)

        with Parallel(n_jobs=self.n_jobs) as parallel:
            for _ in range(self.n_iter):
                # Eltitism
                elitists = sorted(self.population_, key=lambda i: i.fitness_, reverse=True)[:self.n_elitists_]

                # Selection
                n_parents = self.population_size - self.n_elitists_
                parents = self.selection(population=self.population_, n=n_parents, random_state=self.random_state_)
                # Note that this expression swallows the last element, if `population_size` is odd
                parent_pairs = map(lambda *x: x, *([iter(parents)] * 2))

                # Crossover
                children = flatten([(self.crossover(A, B, random_state=self.random_state_),
                                     self.crossover(B, A, random_state=self.random_state_))
                                    for A, B in parent_pairs])

                # Mutation
                mutated_children = [self.mutation(child, random_state=self.random_state_) for child in children]

                # Replacement
                self.population_ = elitists
                self.population_.extend(mutated_children)

                self.fit_population(X, y)
