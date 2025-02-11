import numpy as np


from suprb.solution.initialization import SolutionInit, RandomInit
from ..base import PopulationBasedSolutionComposition
from ..base import SolutionArchive
from suprb.solution.fitness import MultiObjectiveSolutionFitness

from .mutation import SolutionMutation, BitFlips
from .selection import SolutionSelection, Tournament
from .crossover import SolutionCrossover, NPoint
from .sorting import fast_non_dominating_sort


class NonDominatedSortingAlgorithm2(PopulationBasedSolutionComposition):

    def __init__(self,
                 n_iter: int = 32,
                 population_size: int = 32,
                 mutation: SolutionMutation = BitFlips(),
                 crossover: SolutionCrossover = NPoint(n=3),
                 selection: SolutionSelection = Tournament(),
                 mutation_rate: float = 0.025,
                 crossover_rate: float = 0.75,
                 init: SolutionInit = RandomInit(fitness=MultiObjectiveSolutionFitness()),
                 archive: SolutionArchive = SolutionArchive(),
                 random_state: int = None,
                 n_jobs: int = 1,
                 warm_start: bool = True,):
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

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        self.fit_population(X, y)

        for _ in range(self.n_iter):
            fitness_values = np.array([solution.fitness_ for solution in self.archive.population_])
            pareto_ranks = fast_non_dominating_sort(fitness_values)


            self.fit_population(X, y)

