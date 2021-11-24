import numpy as np
from joblib import Parallel, delayed

from suprb2.individual.initialization import IndividualInit, RandomInit
from .crossover import IndividualCrossover, NPoint
from .mutation import IndividualMutation, BitFlips
from .selection import IndividualSelection, Ranking
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

    def __init__(self,
                 n_iter: int = 128,
                 population_size: int = 128,
                 mutation: IndividualMutation = BitFlips(),
                 crossover: IndividualCrossover = NPoint(),
                 selection: IndividualSelection = Ranking(),
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

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        self.fit_population(X, y)

        with Parallel(n_jobs=self.n_jobs) as parallel:
            for _ in range(self.n_iter):
                # Selection
                self.population_ = self.selection(self.population_, self.random_state_)

                # Crossover
                selected = [list(self.random_state_.choice(self.population_, size=2, replace=False)) for _ in
                            range(self.population_size - len(self.population_))]
                children = parallel(delayed(self.crossover)(*parents, self.random_state_) for parents in selected)

                # Mutation
                mutated_children = parallel(delayed(self.mutation)(child, self.random_state_) for child in children)

                # Refit the children
                mutated_children = [child.fit(X, y) for child in mutated_children]

                # Insert the children
                self.population_.extend(mutated_children)
