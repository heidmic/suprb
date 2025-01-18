import numpy as np

from suprb.solution.initialization import SolutionInit, RandomInit
from suprb.utils import flatten
from .crossover import SolutionCrossover, NPoint
from .mutation import SolutionMutation, BitFlips
from .selection import SolutionSelection, Tournament
from ..archive import SolutionArchive, Elitist
from ..base import PopulationBasedSolutionComposition
from suprb.solution import Solution


class SelfAdaptingGeneticAlgorithm(PopulationBasedSolutionComposition):
    """A simple self adapting Genetic Algorithm, implemented acording to 10.1007/s00521-018-3438-9 .

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
    fitness_variance_min: float
    fitness_variance_max: float
    old_gen: list[Solution]
    fitness_mean: float
    fitness_min: float
    fitness_max: float

    def __init__(
        self,
        n_iter: int = 32,
        population_size: int = 32,
        elitist_ratio: float = 0.17,
        mutation: SolutionMutation = BitFlips(),
        crossover: SolutionCrossover = NPoint(n=3),
        selection: SolutionSelection = Tournament(),
        mutation_rate_min: float = 0.01,
        mutation_rate_max: float = 0.1,
        crossover_rate_min: float = 0.4,
        crossover_rate_max: float = 0.9,
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

        self.mutation_rate_min = mutation_rate_min
        self.mutation_rate_max = mutation_rate_max
        self.mutation_rate_current_max = mutation_rate_max
        self.crossover_rate_min = crossover_rate_min
        self.crossover_rate_current_min = crossover_rate_min
        self.crossover_rate_max = crossover_rate_max
        self.mutation = mutation
        self.crossover = crossover
        self.selection = selection
        self.elitist_ratio = elitist_ratio

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        assert self.population_size % 2 == 0

        self.fit_population(X, y)

        self.n_elitists_ = int(self.population_size * self.elitist_ratio)

        fitness_variance = np.var([i.fitness_ for i in self.population_])
        self.fitness_variance_min = fitness_variance
        self.fitness_variance_max = fitness_variance
        self.old_gen = self.population_
        self.fitness_mean = np.mean([i.fitness_ for i in self.population_])
        self.fitness_min = np.min([i.fitness_ for i in self.population_])
        self.fitness_max = np.max([i.fitness_ for i in self.population_])

        for iter_ in range(self.n_iter):

            # Elitism
            elitists = sorted(self.population_, key=lambda i: i.fitness_, reverse=True)[
                : self.n_elitists_
            ]

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
                            self.crossover(
                                A,
                                B,
                                self.crossover_rate_current_min,
                                self.crossover_rate_max,
                                self.fitness_mean,
                                self.fitness_min,
                                self.fitness_max,
                                random_state=self.random_state_,
                            ),
                            self.crossover(
                                B,
                                A,
                                self.crossover_rate_current_min,
                                self.crossover_rate_max,
                                self.fitness_mean,
                                self.fitness_min,
                                self.fitness_max,
                                random_state=self.random_state_,
                            ),
                        )
                        for A, B in parent_pairs
                    ]
                )
            )

            # Mutation
            for child in children:
                child.fit(X, y)
            mutated_children = [
                self.mutation(
                    child,
                    self.mutation_rate_min,
                    self.mutation_rate_current_max,
                    self.fitness_mean,
                    self.fitness_max,
                    random_state=self.random_state_,
                )
                for child in children
            ]

            # Keep copy of old generation for comparison
            self.old_gen = self.population_

            # Replacement
            self.population_ = elitists
            self.population_.extend(mutated_children)

            self.fit_population(X, y)
            self.fitness_mean = np.mean(([i.fitness_ for i in self.population_]))
            self.fitness_min = np.min(([i.fitness_ for i in self.population_]))
            self.fitness_max = np.max(([i.fitness_ for i in self.population_]))

            # Adjust Rates for current fitness
            if iter_ > 0:
                self.adjust_rates()

    def calc_similarity(self):
        try:
            dot_sum = np.sum(
                [
                    np.multiply(self.old_gen[i].fitness_, self.population_[i].fitness_)
                    for i in range(len(self.population_))
                ]
            )
            length_new = np.sqrt(
                np.sum([np.square(i.fitness_) for i in self.population_]) + np.exp(-10)
            )
            length_old = np.sqrt(
                np.sum([np.square(i.fitness_) for i in self.old_gen]) + np.exp(-10)
            )
            cosine_similarity = dot_sum / (length_new + length_old)
            genome_similarity = (
                np.intersect1d(
                    [i.fitness_ for i in self.population_],
                    [i.fitness_ for i in self.old_gen],
                ).size
                / np.union1d(
                    [i.fitness_ for i in self.population_],
                    [i.fitness_ for i in self.old_gen],
                ).size
            )
        except ZeroDivisionError:
            return 1
        return cosine_similarity * genome_similarity

    def calc_diversity(self):
        fitness_variance = np.var([i.fitness_ for i in self.population_])
        if fitness_variance > self.fitness_variance_max:
            self.fitness_variance_max = fitness_variance
        if fitness_variance < self.fitness_variance_min:
            self.fitness_variance_min = fitness_variance
        a = (fitness_variance - self.fitness_variance_min) / self.fitness_variance_max
        if self.fitness_max == self.fitness_min:
            b = 0
        else:
            b = np.cos(
                np.pi
                / 2
                * (
                    1
                    - (self.fitness_mean - self.fitness_min)
                    / (self.fitness_max - self.fitness_min)
                )
            )
        return a * b

    def calc_quality(self):
        return self.calc_diversity() * (1 - self.calc_similarity())

    def adjust_rates(self):
        quality = self.calc_quality()
        # crossover rate
        self.crossover_rate_current_min = (
            self.crossover_rate_min
            + quality * (self.crossover_rate_max - self.crossover_rate_min) / 2
        )
        # mutationrate
        self.mutation_rate_current_max = (
            self.mutation_rate_max
            - quality * (self.mutation_rate_max - self.mutation_rate_min) / 2
        )
