import numpy as np


from suprb.optimizer.solution.ga.mutation import SolutionMutation, BitFlips
from suprb.optimizer.solution.ga.crossover import NPoint, SolutionCrossover, SelfAdaptiveCrossover
from suprb.optimizer.solution.ga.selection import SolutionSelection, Tournament, Ageing
from suprb.optimizer.solution.saga.utils import SagaSolution, SagaRandomInit

from suprb.solution.initialization import SolutionInit, RandomInit, Solution

from suprb.utils import flatten

from ..archive import Elitist, SolutionArchive
from ..base import PopulationBasedSolutionComposition
from suprb.utils import RandomState


class SelfAdaptingGeneticAlgorithmBase(PopulationBasedSolutionComposition):
    """A simple self adapting Genetic Algorithm, implemented acording to 10.1109/20.952626 .

    Parameters
    ----------
    n_iter: int
        Iterations the the metaheuristic will perform.
    population_size: int
        Number of solutions in the population.
    elitist_ratio: float
        Ratio of elitist to population size
    mutation: SolutionMutation
        Class used for mutation
    mutation_rate: float
        Mutation rate for Rules
    crossover: SolutionCrossover
        Class used for crossover
    crossover_rate: float
        Crossover rate for Rules
    selection: SolutionSelection
        Class used for solution selection
    init: SolutionInit
        Class used for solution initialization
    archive: SolutionArchive
        Class used for solution archive
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
    n_jobs: int
        The number of threads / processes the optimization uses.
    warm_start: bool
        If False, solutions are generated new for every `optimize()` call.
        If True, solutions are used from previous runs.
    """

    n_elitists_: int

    def __init__(
        self,
        n_iter: int,
        population_size: int,
        elitist_ratio: float,
        mutation: SolutionMutation,
        mutation_rate: float,
        crossover: SolutionCrossover,
        crossover_rate: float,
        selection: SolutionSelection,
        init: SolutionInit,
        archive: SolutionArchive,
        random_state: int,
        n_jobs: int,
        warm_start: bool,
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
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def fitness_calculation(self):
        pass

    def update_genetic_operator_rates(self):
        pass

    def crossover_children(self, parent_pairs, crossover_rate):
        # fmt: off
        return list(flatten(
            [(self.crossover(A, B, crossover_rate, self.random_state_), self.crossover(B, A, crossover_rate, self.random_state_)) 
             for A, B in parent_pairs
        ]))
        # fmt: on

    def mutate_children(self, children, X=None, y=None):
        return [self.mutation(child, self.mutation_rate, random_state=self.random_state_) for child in children]

    def parent_selection(self):
        return self.selection(population=self.population_, n=self.population_size, random_state=self.random_state_)

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        assert self.population_size % 2 == 0
        self.fit_population(X, y)

        self.n_elitists_ = int(self.population_size * self.elitist_ratio)

        for _ in range(self.n_iter):
            self.fitness_calculation()
            self.update_genetic_operator_rates()
            elitists = sorted(self.population_, key=lambda i: i.fitness_, reverse=True)[: self.n_elitists_]
            parents = self.parent_selection()

            # Note that this expression swallows the last element, if `population_size` is odd
            parent_pairs = map(lambda *x: x, *([iter(parents)] * 2))

            children = self.crossover_children(parent_pairs, self.crossover_rate)
            mutated_children = self.mutate_children(children, X, y)

            self.population_ = elitists
            self.population_.extend(mutated_children)

            self.fit_population(X, y)


class SelfAdaptingGeneticAlgorithm1(SelfAdaptingGeneticAlgorithmBase):
    """
    Parameters
    ----------
    mutation_rate_multiplier: float
        Multiplier for mutation rate
    crossover_rate_multiplier: float
        Multiplier for crossover rate
    v_min: float
        Lower bound for the population diversity, where a higher number means less diverse.
    v_max: float
        Upper bound for the population diversity, where a higher number means less diverse.
    mutation_rate_min: float
        Lower bound for mutation rate
    mutation_rate_max: float
        Upper bound for mutation rate
    crossover_rate_min: float
        Lower bound for crossover rate
    crossover_rate_max: float
        Upper bound for crossover rate
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
        v_min: float = 0.005,
        v_max: float = 0.15,
        mutation_rate: float = 0.025,
        mutation_rate_min: float = 0.001,
        mutation_rate_max: float = 0.25,
        mutation_rate_multiplier=1.1,
        crossover_rate: float = 0.75,
        crossover_rate_min: float = 0.5,
        crossover_rate_max: float = 1.0,
        crossover_rate_multiplier=1.1,
        init: SolutionInit = RandomInit(),
        archive: SolutionArchive = Elitist(),
        random_state: int = None,
        n_jobs: int = 1,
        warm_start: bool = True,
    ):
        super().__init__(
            n_iter=n_iter,
            population_size=population_size,
            elitist_ratio=elitist_ratio,
            crossover=crossover,
            selection=selection,
            mutation=mutation,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            init=init,
            archive=archive,
            random_state=random_state,
            n_jobs=n_jobs,
            warm_start=warm_start,
        )

        self.mutation_rate_multiplier = mutation_rate_multiplier
        self.crossover_rate_multiplier = crossover_rate_multiplier

        self.v_min = v_min
        self.v_max = v_max

        self.mutation_rate_min = mutation_rate_min
        self.mutation_rate_max = mutation_rate_max
        self.crossover_rate_min = crossover_rate_min
        self.crossover_rate_max = crossover_rate_max

    def update_genetic_operator_rates(self):
        gdm = np.mean([i.fitness_ for i in self.population_]) / np.max([i.fitness_ for i in self.population_])
        if gdm > self.v_max:
            self.mutation_rate = min(self.mutation_rate_max, self.mutation_rate * self.mutation_rate_multiplier)
            self.crossover_rate = max(self.crossover_rate_min, self.crossover_rate / self.crossover_rate_multiplier)
        elif gdm < self.v_min:
            self.mutation_rate = max(self.mutation_rate_min, self.mutation_rate / self.mutation_rate_multiplier)
            self.crossover_rate = min(self.crossover_rate_max, self.crossover_rate * self.crossover_rate_multiplier)


class SelfAdaptingGeneticAlgorithm2(SelfAdaptingGeneticAlgorithmBase):
    """
    Parameters
    ----------
    mutation_rate_min: float
        Lower bound for mutation rate
    mutation_rate_max: float
        Upper bound for mutation rate
    crossover_rate_min: float
        Lower bound for crossover rate
    crossover_rate_max: float
        Upper bound for crossover rate
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
            elitist_ratio=elitist_ratio,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            mutation_rate=None,
            crossover_rate=None,
            init=init,
            archive=archive,
            random_state=random_state,
            n_jobs=n_jobs,
            warm_start=warm_start,
        )

        self.mutation_rate_min = mutation_rate_min
        self.mutation_rate_max = mutation_rate_max
        self.crossover_rate_min = crossover_rate_min
        self.crossover_rate_max = crossover_rate_max

    def fitness_calculation(self):
        fitness_variance = np.var([i.fitness_ for i in self.population_])
        self.fitness_variance_min = fitness_variance
        self.fitness_variance_max = fitness_variance
        self.old_gen = self.population_
        self.fitness_mean = np.mean([i.fitness_ for i in self.population_])
        self.fitness_min = np.min([i.fitness_ for i in self.population_])
        self.fitness_max = np.max([i.fitness_ for i in self.population_])

    def mutate_func(self, solution: Solution) -> Solution:
        if self.fitness_max == self.fitness_mean:
            mutation_rate = self.mutation_rate_min
        elif solution.fitness_ > self.fitness_mean:
            mutation_rate = self.mutation_rate_min + (self.mutation_rate_current_max - self.mutation_rate_min) * (
                (self.fitness_max - solution.fitness_) / (self.fitness_max - self.fitness_mean)
            )
        else:
            mutation_rate = self.mutation_rate_current_max

        bit_flips = self.random_state_.random(solution.genome.shape) < mutation_rate
        genome = np.logical_xor(solution.genome, bit_flips)

        return solution.clone(genome=genome)

    def mutate_children(self, children, X, y):
        for child in children:
            child.fit(X, y)

        mutated_children = [self.mutate_func(child) for child in children]

        return mutated_children

    def crossover_func(self, A: Solution, B: Solution) -> Solution:
        fitness_parents_mean = (A.fitness_ + B.fitness_) / 2

        rate_diff = self.crossover_rate_max - self.crossover_rate_min
        fitness_diff = abs(self.fitness_mean - fitness_parents_mean)

        if self.fitness_mean in {self.fitness_min, self.fitness_max}:
            new_rate = self.crossover_rate_max
        elif fitness_parents_mean <= self.fitness_mean:
            new_rate = self.crossover_rate_max - rate_diff * (fitness_diff / (self.fitness_mean - self.fitness_min))
        else:
            new_rate = self.crossover_rate_max - rate_diff * (fitness_diff / (self.fitness_max - self.fitness_mean))

        if self.random_state_.random() < new_rate:
            return self.crossover._crossover(A=A, B=B, random_state=self.random_state_)

        return A

    def crossover_children(self, parent_pairs, crossover_rate):
        return list(flatten([(self.crossover_func(A, B), self.crossover_func(B, A)) for A, B in parent_pairs]))

    def calc_similarity(self):
        try:
            # fmt: off
            dot_sum = np.sum([np.multiply(self.old_gen[i].fitness_, self.population_[i].fitness_)
                                for i in range(len(self.population_))])
            # fmt: on

            length_new = np.sqrt(np.sum([np.square(i.fitness_) for i in self.population_]) + np.exp(-10))
            length_old = np.sqrt(np.sum([np.square(i.fitness_) for i in self.old_gen]) + np.exp(-10))
            cosine_similarity = dot_sum / (length_new + length_old)
            intersect_1d = np.intersect1d([i.fitness_ for i in self.population_], [i.fitness_ for i in self.old_gen])
            union_1d = np.union1d([i.fitness_ for i in self.population_], [i.fitness_ for i in self.old_gen])
            genome_similarity = intersect_1d.size / union_1d.size
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
            b = np.cos(np.pi / 2 * (1 - (self.fitness_mean - self.fitness_min) / (self.fitness_max - self.fitness_min)))

        return a * b

    def calc_quality(self):
        return self.calc_diversity() * (1 - self.calc_similarity())

    def update_genetic_operator_rates(self):
        quality = self.calc_quality()

        # fmt: off
        self.crossover_rate_current_min = (self.crossover_rate_min + quality * (self.crossover_rate_max - self.crossover_rate_min) / 2)
        self.mutation_rate_current_max = (self.mutation_rate_max - quality * (self.mutation_rate_max - self.mutation_rate_min) / 2)
        # fmt: on


class SelfAdaptingGeneticAlgorithm3(SelfAdaptingGeneticAlgorithmBase):
    """
    Parameters
    ----------
    parameter_mutation_rate: float
        Rate for doing crossover and mutation on rules
    """

    n_elitists_: int
    population_: list[SagaSolution]

    def __init__(
        self,
        n_iter: int = 32,
        population_size: int = 32,
        elitist_ratio: float = 0.17,
        mutation: SolutionMutation = BitFlips(),
        crossover: SelfAdaptiveCrossover = SelfAdaptiveCrossover(parameter_mutation_rate=0.05),
        selection: SolutionSelection = Tournament(),
        init: SolutionInit = SagaRandomInit(),
        archive: SolutionArchive = Elitist(),
        random_state: int = None,
        n_jobs: int = 1,
        warm_start: bool = True,
        parameter_mutation_rate=0.05,
    ):
        super().__init__(
            n_iter=n_iter,
            population_size=population_size,
            elitist_ratio=elitist_ratio,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            mutation_rate=parameter_mutation_rate,
            crossover_rate=parameter_mutation_rate,
            init=init,
            archive=archive,
            random_state=random_state,
            n_jobs=n_jobs,
            warm_start=warm_start,
        )

        self.parameter_mutation_rate = parameter_mutation_rate

    def crossover_children(self, parent_pairs, crossover_rate):
        # fmt: off
        return list(flatten(
            [(self.crossover(A, B, random_state=self.random_state_), self.crossover(B, A, random_state=self.random_state_)) 
             for A, B in parent_pairs
        ]))
        # fmt: on

    def mutate_children(self, children, X=None, y=None):
        return children


class SasGeneticAlgorithm(SelfAdaptingGeneticAlgorithmBase):
    """
    Parameters
    ----------
    initial_population_size: The initialize population size
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
        archive: SolutionArchive = Elitist(),
        random_state: int = None,
        n_jobs: int = 1,
        warm_start: bool = True,
        mutation_rate: float = 0.001,
        crossover_rate: float = 0.9,
    ):
        super().__init__(
            n_iter=n_iter,
            population_size=32,
            elitist_ratio=0,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            init=init,
            archive=archive,
            random_state=random_state,
            n_jobs=n_jobs,
            warm_start=warm_start,
        )

        self.initial_population_size = initial_population_size

    def parent_selection(self):
        return self.selection(
            population=self.population_,
            initial_population_size=self.initial_population_size,
            random_state=self.random_state_,
        )
