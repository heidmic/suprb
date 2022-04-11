import numpy as np

from suprb.solution import SolutionInit
from suprb.solution.initialization import RandomInit
from ..archive import Elitist, SolutionArchive
from ..base import PopulationBasedSolutionComposition


class RandomSearch(PopulationBasedSolutionComposition):
    """ A simple Random Search.
    Note that this is not really population-based, and only used to shorten the implementation.

    Parameters
    ----------
    n_iter: int
        Iterations the metaheuristic will perform.
    population_size: int
        Number of solutions in the population.
    init: SolutionInit
    archive: Archive
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
    warm_start: bool
        If False, solutions are generated new for every `optimize()` call.
        If True, solutions are used from previous runs.
    n_jobs: int
        The number of threads / processes the optimization uses.
    """

    def __init__(self,
                 n_iter: int = 128,
                 population_size: int = 128,
                 init: SolutionInit = RandomInit(),
                 archive: SolutionArchive = Elitist(),
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

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        self.fit_population(X, y)

        new_population = []
        for _ in range(self.population_size):
            solutions = [self.init(self.pool_, self.random_state_).fit(X, y)
                         for _ in range(self.n_iter)]
            new_population.append(max(solutions, key=lambda i: i.fitness_))

        self.population_ = new_population
