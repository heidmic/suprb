import numpy as np

from suprb.solution import SolutionInit
from suprb.solution.initialization import RandomInit
from .builder import SolutionBuilder, Complete
from .pheromones import PheromoneUpdate, Fitness
from .selection import AntSelection, NBest
from ..archive import Elitist, SolutionArchive
from ..base import PopulationBasedSolutionComposition


class AntColonyOptimization(PopulationBasedSolutionComposition):
    """Ant Colony Optimization written in Python.

    The base version was taken from https://doi.org/10/dq339r.

    Parameters
    ----------
    n_iter: int
        Iterations the metaheuristic will perform.
    population_size: int
        Number of solutions in the population.
    evaporation_rate: float
        Evaporation of the pheromones.
    max_pheromone: float
    min_pheromone: float
    builder: Builder
    selection: AntSelection
    pheromones: Pheromones
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

    pheromone_matrix_: np.ndarray

    def __init__(
        self,
        n_iter: int = 32,
        population_size: int = 32,
        evaporation_rate: float = 0.78,
        max_pheromone: float = 10,
        min_pheromone: float = 0.1,
        builder: SolutionBuilder = Complete(),
        selection: AntSelection = NBest(),
        pheromones: PheromoneUpdate = Fitness(),
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

        self.evaporation_rate = evaporation_rate
        self.max_pheromone = max_pheromone
        self.min_pheromone = min_pheromone

        self.builder = builder
        self.selection = selection
        self.pheromones = pheromones

    def _init_pheromone_matrix(self):
        """Initialize an empty matrix or pad the matrix for additional rules."""

        if not self.warm_start or not hasattr(self, "pheromone_matrix_"):
            self.pheromone_matrix_ = self.builder.pad_pheromone_matrix(
                None, len(self.pool_)
            )
        else:
            self.pheromone_matrix_ = self.builder.pad_pheromone_matrix(
                self.pheromone_matrix_, len(self.pool_)
            )

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        self._init_pheromone_matrix()
        self.fit_population(X, y)

        for _ in range(self.n_iter):

            # Construct ants and evaluate
            self.population_ = [
                self.builder(
                    solution=solution,
                    pheromones=self.pheromone_matrix_,
                    pool=self.pool_,
                    random_state=self.random_state_,
                )
                for solution in self.population_
            ]
            self.fit_population(X, y)

            # Evaporation
            self.pheromone_matrix_ *= 1 - self.evaporation_rate

            # Select ants to update the pheromones with and update the pheromones
            selected = self.selection(self.population_, random_state=self.random_state_)
            for ant in selected:
                self.builder.update_pheromones(
                    ant,
                    pheromones=self.pheromone_matrix_,
                    delta_tau=self.pheromones(ant),
                )

            # Clip pheromones
            self.pheromone_matrix_ = np.clip(
                self.pheromone_matrix_, self.min_pheromone, self.max_pheromone
            )
