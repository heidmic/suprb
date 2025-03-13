import numpy as np


from suprb.optimizer.solution.ga.mutation import SolutionMutation, BitFlips
from suprb.optimizer.solution.ga.crossover import NPoint, SolutionCrossover, SagaCrossover
from suprb.optimizer.solution.ga.selection import SolutionSelection, Tournament, Ageing
from suprb.optimizer.solution.saga.utils import SagaSolution, SagaElitist, SagaRandomInit

from suprb.solution.initialization import SolutionInit, RandomInit, Solution

from suprb.utils import flatten

from ..archive import Elitist, SolutionArchive
from ..base import PopulationBasedSolutionComposition
from suprb.utils import RandomState


class SelfAdaptingGeneticAlgorithm1(PopulationBasedSolutionComposition):
    """A simple self adapting Genetic Algorithm, implemented acording to 10.1109/20.952626 .

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

    def adjust_rates(self):
        gdm = np.mean([i.fitness_ for i in self.population_]) / np.max([i.fitness_ for i in self.population_])
        if gdm > self.v_max:
            self.mutation_rate = min(self.mutation_rate_max, self.mutation_rate * self.mutation_rate_multiplier)
            self.crossover_rate = max(self.crossover_rate_min, self.crossover_rate / self.crossover_rate_multiplier)
        elif gdm < self.v_min:
            self.mutation_rate = max(self.mutation_rate_min, self.mutation_rate / self.mutation_rate_multiplier)
            self.crossover_rate = min(self.crossover_rate_max, self.crossover_rate * self.crossover_rate_multiplier)

    def crossover_children(self, parent_pairs):
        return list(
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

    def mutate_children(self, children):
        return [self.mutation(child, self.mutation_rate, random_state=self.random_state_) for child in children]

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        assert self.population_size % 2 == 0
        self.fit_population(X, y)

        self.n_elitists_ = int(self.population_size * self.elitist_ratio)

        for _ in range(self.n_iter):
            # Adjust Rates for current fitness
            self.adjust_rates()

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
            children = self.crossover_children(parent_pairs)

            # Mutation
            mutated_children = self.mutate_children(children)

            # Replacement
            self.population_ = elitists
            self.population_.extend(mutated_children)

            self.fit_population(X, y)


class SelfAdaptingGeneticAlgorithm2(PopulationBasedSolutionComposition):
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

    def fitness_calculation(self):
        fitness_variance = np.var([i.fitness_ for i in self.population_])
        self.fitness_variance_min = fitness_variance
        self.fitness_variance_max = fitness_variance
        self.old_gen = self.population_
        self.fitness_mean = np.mean([i.fitness_ for i in self.population_])
        self.fitness_min = np.min([i.fitness_ for i in self.population_])
        self.fitness_max = np.max([i.fitness_ for i in self.population_])

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        assert self.population_size % 2 == 0
        self.fit_population(X, y)

        self.n_elitists_ = int(self.population_size * self.elitist_ratio)

        for _ in range(self.n_iter):
            self.fitness_calculation()

            # Adjust Rates for current fitness
            self.adjust_rates()

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
            children = self.crossover_children(parent_pairs)

            # Mutation
            mutated_children = self.mutate_children(X, y, children)

            # Replacement
            self.population_ = elitists
            self.population_.extend(mutated_children)

            self.fit_population(X, y)

    def saga2_mutation(
        self,
        solution: Solution,
        mutation_rate_min: float,
        mutation_rate_max: float,
        fitness_mean: float,
        fitness_max: float,
        random_state: RandomState,
    ) -> Solution:
        if fitness_max == fitness_mean:
            mutation_rate = mutation_rate_min
        elif solution.fitness_ > fitness_mean:
            mutation_rate = mutation_rate_min + (mutation_rate_max - mutation_rate_min) * (
                (fitness_max - solution.fitness_) / (fitness_max - fitness_mean)
            )
        else:
            mutation_rate = mutation_rate_max

        bit_flips = random_state.random(solution.genome.shape) < mutation_rate
        genome = np.logical_xor(solution.genome, bit_flips)

        return solution.clone(genome=genome)

    def saga2_crossover(
        self,
        A: Solution,
        B: Solution,
        crossover_rate_min,
        crossover_rate_max,
        fitness_mean,
        fitness_min,
        fitness_max,
        random_state: RandomState,
    ) -> Solution:
        fitness_parents_mean = (A.fitness_ + B.fitness_) / 2

        crossover_rate_diff = crossover_rate_max - crossover_rate_min
        fitness_diff = abs(fitness_mean - fitness_parents_mean)

        if fitness_mean in {fitness_min, fitness_max}:
            crossover_rate = crossover_rate_max
        elif fitness_parents_mean <= fitness_mean:
            crossover_rate = crossover_rate_max - crossover_rate_diff * (fitness_diff / (fitness_mean - fitness_min))
        else:
            crossover_rate = crossover_rate_max - crossover_rate_diff * (fitness_diff / (fitness_max - fitness_mean))

        if random_state.random() < crossover_rate:
            return self.crossover._crossover(A=A, B=B, random_state=random_state)

        return A

    def crossover_children(self, parent_pairs):
        return list(
            flatten(
                [
                    (
                        self.saga2_crossover(
                            A,
                            B,
                            self.crossover_rate_current_min,
                            self.crossover_rate_max,
                            self.fitness_mean,
                            self.fitness_min,
                            self.fitness_max,
                            random_state=self.random_state_,
                        ),
                        self.saga2_crossover(
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

    def mutate_children(self, X, y, children):
        for child in children:
            child.fit(X, y)

        mutated_children = [
            self.saga2_mutation(
                child,
                self.mutation_rate_min,
                self.mutation_rate_current_max,
                self.fitness_mean,
                self.fitness_max,
                random_state=self.random_state_,
            )
            for child in children
        ]

        return mutated_children

    def calc_similarity(self):
        try:
            dot_sum = np.sum(
                [
                    np.multiply(self.old_gen[i].fitness_, self.population_[i].fitness_)
                    for i in range(len(self.population_))
                ]
            )
            length_new = np.sqrt(np.sum([np.square(i.fitness_) for i in self.population_]) + np.exp(-10))
            length_old = np.sqrt(np.sum([np.square(i.fitness_) for i in self.old_gen]) + np.exp(-10))
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
            b = np.cos(np.pi / 2 * (1 - (self.fitness_mean - self.fitness_min) / (self.fitness_max - self.fitness_min)))
        return a * b

    def calc_quality(self):
        return self.calc_diversity() * (1 - self.calc_similarity())

    def adjust_rates(self):
        quality = self.calc_quality()
        # crossover rate
        self.crossover_rate_current_min = (
            self.crossover_rate_min + quality * (self.crossover_rate_max - self.crossover_rate_min) / 2
        )
        # mutationrate
        self.mutation_rate_current_max = (
            self.mutation_rate_max - quality * (self.mutation_rate_max - self.mutation_rate_min) / 2
        )


class SelfAdaptingGeneticAlgorithm3(PopulationBasedSolutionComposition):
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
                            self.crossover(A, B, crossover_rate=self.crossover_rate, random_state=self.random_state_),
                            self.crossover(B, A, crossover_rate=self.crossover_rate, random_state=self.random_state_),
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
