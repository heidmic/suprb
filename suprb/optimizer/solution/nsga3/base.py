
import numpy as np

from suprb import Solution
from suprb.solution.initialization import SolutionInit, RandomInit
from ..base import MOSolutionComposition
from suprb.solution.fitness import BasicMOSolutionFitness
from suprb.utils import flatten


from .mutation import SolutionMutation, BitFlips
from .selection import SolutionSelection, BinaryTournament
from .crossover import SolutionCrossover, NPoint
from .sorting import fast_non_dominated_sort
from .reference import das_dennis_points, calc_ref_direction_distances
from ..sampler import SolutionSampler, NormalSolutionSampler


class NonDominatedSortingGeneticAlgorithm3(MOSolutionComposition):
    """A fast and elitist multiobjective genetic algorithm.

    Implemented as described in 10.1007/978-3-319-15892-1_3

    Parameters
    ----------
    n_iter: int
        Iterations the metaheuristic will perform.
    population_size: int
        Number of solutions in the population.
    mutation: SolutionMutation
    crossover: SolutionCrossover
    selection: SolutionSelection
    init: SolutionInit
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
    warm_start: bool
        If False, solutions are generated new for every `optimize()` call.
        If True, solutions are used from previous runs.
    n_jobs: int
        The number of threads / processes the optimization uses.
    """
    def __init__(self,
                 n_iter: int = 32,
                 n_reference_points: int = 15,
                 population_size: int = 32,
                 mutation: SolutionMutation = BitFlips(),
                 crossover: SolutionCrossover = NPoint(n=3),
                 selection: SolutionSelection = BinaryTournament(),
                 sampler: SolutionSampler = NormalSolutionSampler(),
                 mutation_rate: float = 0.025,
                 crossover_rate: float = 0.75,
                 init: SolutionInit = RandomInit(fitness=BasicMOSolutionFitness()),
                 random_state: int = None,
                 n_jobs: int = 1,
                 warm_start: bool = True,):
        super().__init__(
            n_iter=n_iter,
            population_size=population_size,
            init=init,
            archive=None,
            random_state=random_state,
            n_jobs=n_jobs,
            warm_start=warm_start,
        )

        self.n_reference_points = n_reference_points
        self.mutation = mutation
        self.crossover = crossover
        self.selection = selection
        self.sampler = sampler

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        self.fit_population(X, y)
        reference_points = das_dennis_points(self.n_reference_points, 2)

        for _ in range(self.n_iter):
            fitness_values = np.array([solution.fitness_ for solution in self.population_])
            pareto_ranks = fast_non_dominated_sort(fitness_values)
            closest_ref_direction, ref_direction_distance = calc_ref_direction_distances(fitness_values,
                                                                                         reference_points)


            parents = self.selection(
                population=self.population_,
                n=self.population_size,
                random_state=self.random_state_,
                pareto_ranks=pareto_ranks,
                closest_ref_direction=closest_ref_direction,
                ref_direction_distance=ref_direction_distance
            )

            # Note that this expression swallows the last element, if `population_size` is odd
            parent_pairs = map(lambda *x: x, *([iter(parents)] * 2))

            # Crossover
            children = list(
                flatten(
                    [
                        (
                            self.crossover(A, B, random_state=self.random_state_),
                            self.crossover(B, A, random_state=self.random_state_),
                        )
                        for A, B in parent_pairs
                    ]
                )
            )
            # If `population_size` is odd, we add the solution not selected for reproduction directly
            if self.population_size % 2 != 0:
                children.append(parents[-1])

            # Mutation
            mutated_children = [self.mutation(child, random_state=self.random_state_).fit(X, y) for child in children]
            intermediate_pop = self.population_ + mutated_children

            # Selecting the next generation by the elitist of pareto rank and crowding score
            intermediate_fitness_values = np.array([solution.fitness_ for solution in intermediate_pop])
            intermediate_pareto_ranks = fast_non_dominated_sort(intermediate_fitness_values)
            intermediate_closest_ref, intermediate_ref_dist = calc_ref_direction_distances(intermediate_fitness_values,
                                                                  intermediate_pareto_ranks)


    def pareto_front(self) -> list[Solution]:
        if not hasattr(self, "population_") or not self.population_:
            return []
        fitness_values = np.array([solution.fitness_ for solution in self.population_])
        pareto_ranks = fast_non_dominated_sort(fitness_values)
        pareto_front = np.array(self.population_)[pareto_ranks == 0]
        return pareto_front
