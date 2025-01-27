import numpy as np

from suprb.solution import SolutionInit
from suprb.solution.initialization import RandomInit
from .position import SolutionPositionUpdate, Sigmoid
from ..archive import Elitist, SolutionArchive
from ..base import PopulationBasedSolutionComposition


class GreyWolfOptimizer(PopulationBasedSolutionComposition):
    """Grey Wolf Optimizer written in Python.

    The base version was taken from https://doi.org/10/gfxvkw.

    Parameters
    ----------
    n_iter: int
        Iterations the metaheuristic will perform.
    population_size: int
        Number of solutions in the population.
    position: SolutionPositionUpdate
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

    def __init__(
        self,
        n_iter: int = 32,
        population_size: int = 32,
        n_leaders: int = 2,
        position: SolutionPositionUpdate = Sigmoid(),
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

        self.n_leaders = n_leaders
        self.position = position

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        self.fit_population(X, y)

        a = 2
        step_size = 2 / self.n_iter

        for _ in range(self.n_iter):
            # Get Alpha, Beta, and Delta
            leaders = sorted(self.population_, key=lambda i: i.fitness_, reverse=True)[: self.n_leaders]

            # Update the positions of all wolves
            self.population_ = self.position(
                leaders=leaders,
                population=self.population_,
                a=a,
                random_state=self.random_state_,
            )

            # Update a
            a -= step_size

            self.fit_population(X, y)
