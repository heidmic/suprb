import numpy as np

from suprb.solution.initialization import SolutionInit, RandomInit
from suprb.utils import flatten
from .crossover import SolutionCrossover, NPoint
from .mutation import SolutionMutation, BitFlips
from .selection import SolutionSelection, Tournament
from ..archive import SolutionArchive, Elitist
from ..base import PopulationBasedSolutionComposition


class SelfAdaptingGeneticAlgorithm(PopulationBasedSolutionComposition):
    """ A simple self adapting Genetic Algorithm, implemented acording to 10.1109/20.952626 .

    Parameters
    ----------
    n_iter: int
        Iterations the the metaheuristic will perform.
    population_size: int
        Number of solutions in the population.
    mutation: SolutionMutation
    crossover: SolutionCrossover
    selection: SolutionSelection
    v_min: float
        Lower bound for the population diversity, where a higher number means less diverse.
    v_max: float
        Upper bound for the population diversity, where a higher number means less diverse.
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

    def __init__(self,
                 n_iter: int = 32,
                 population_size: int = 32,
                 elitist_ratio: float = 0.17,
                 mutation: SolutionMutation = BitFlips(),
                 crossover: SolutionCrossover = NPoint(n=3),
                 selection: SolutionSelection = Tournament(),
                 v_min: float = 0.005,
                 v_max: float = 0.15,
                 mutation_rate: float = 0.025,
                 mutation_rate_min: float = 0.001,
                 mutation_rate_max: float = 0.25,
                 mutation_rate_multiplier = 1.1,
                 crossover_rate: float = 0.75,
                 crossover_rate_min: float = 0.5,
                 crossover_rate_max: float = 1.0,
                 crossover_rate_multiplier = 1.1,
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

        self.v_min = v_min
        self.v_max = v_max
        self.mutation_rate = mutation_rate
        self.mutation_rate_min = mutation_rate_min
        self.mutation_rate_max = mutation_rate_max
        self.mutation_rate_multiplier = mutation_rate_multiplier
        self.crossover_rate = crossover_rate
        self.crossover_rate_min = crossover_rate_min
        self.crossover_rate_max = crossover_rate_max
        self.crossover_rate_multiplier = crossover_rate_multiplier
        self.mutation = mutation
        self.crossover = crossover
        self.selection = selection
        self.elitist_ratio = elitist_ratio

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        self.fit_population(X, y)

        self.n_elitists_ = int(self.population_size * self.elitist_ratio)

        for _ in range(self.n_iter):
            # Adjust Rates for current fitness
            self.adjust_rates()

            # Eltitism
            elitists = sorted(self.population_, key=lambda i: i.fitness_, reverse=True)[:self.n_elitists_]

            # Selection
            parents = self.selection(population=self.population_, n=self.population_size,
                                     random_state=self.random_state_)

            # Note that this expression swallows the last element, if `population_size` is odd
            parent_pairs = map(lambda *x: x, *([iter(parents)] * 2))

            # Crossover
            children = list(flatten([(self.crossover(A, B, self.crossover_rate, random_state=self.random_state_),
                                      self.crossover(B, A, self.crossover_rate, random_state=self.random_state_))
                                     for A, B in parent_pairs]))
            # If `population_size` is odd, we add the solution not selected for reproduction directly
            if self.population_size % 2 != 0:
                children.append(parents[-1])

            # Mutation
            mutated_children = [self.mutation(child, self.mutation_rate, random_state=self.random_state_) for child in children]

            # Replacement
            self.population_ = elitists
            self.population_.extend(mutated_children)

            self.fit_population(X, y)


    def calc_gdm(self):
        return np.mean([i.fitness_ for i in self.population_]) / np.max([i.fitness_ for i in self.population_])
    

    def adjust_rates(self):
        gdm = self.calc_gdm()
        if gdm > self.v_max:
            self.mutation_rate = min(self.mutation_rate_max, self.mutation_rate * self.mutation_rate_multiplier)
            self.crossover_rate = max(self.crossover_rate_min, self.crossover_rate / self.crossover_rate_multiplier)
        elif gdm < self.v_min:
            self.mutation_rate = max(self.mutation_rate_min, self.mutation_rate / self.mutation_rate_multiplier)
            self.crossover_rate = min(self.crossover_rate_max, self.crossover_rate * self.crossover_rate_multiplier)
