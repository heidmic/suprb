
import numpy as np

from suprb import Solution
from suprb.solution.initialization import SolutionInit, RandomInit
from ..base import MOSolutionComposition
from suprb.solution.fitness import BasicMOSolutionFitness
from suprb.utils import flatten


from .mutation import SolutionMutation, BitFlips
from .selection import SolutionSelection, ReferenceBasedBinaryTournament
from .crossover import SolutionCrossover, NPoint
from .sorting import fast_non_dominated_sort
from .reference import das_dennis_points, calc_ref_direction_distances
from .normalise import NSGAIIINormaliser, HyperPlaneNormaliser
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
                 n_dimensions: int = 2,
                 population_size: int = 32,
                 normaliser: NSGAIIINormaliser = None,
                 mutation: SolutionMutation = BitFlips(),
                 crossover: SolutionCrossover = NPoint(n=3),
                 selection: SolutionSelection = ReferenceBasedBinaryTournament(),
                 sampler: SolutionSampler = NormalSolutionSampler(),
                 mutation_rate: float = 0.025,
                 crossover_rate: float = 0.75,
                 init: SolutionInit = RandomInit(fitness=BasicMOSolutionFitness()),
                 random_state: int = None,
                 n_jobs: int = 1,
                 warm_start: bool = True, ):
        super().__init__(
            n_iter=n_iter,
            population_size=population_size,
            init=init,
            archive=None,
            sampler=sampler,
            random_state=random_state,
            n_jobs=n_jobs,
            warm_start=warm_start,
        )

        self.n_dimensions = n_dimensions
        self.n_reference_points = n_reference_points
        self.mutation = mutation
        self.crossover = crossover
        self.selection = selection
        self.sampler = sampler
        fitness = self.init.fitness
        if normaliser is None:
            normaliser = HyperPlaneNormaliser(
                    len(fitness.objective_func_), fitness.worst_point_estimate_
            )
        self.normaliser = normaliser
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        self.fit_population(X, y)
        fitness_values = np.array([solution.fitness_ for solution in self.population_])
        pareto_ranks = fast_non_dominated_sort(fitness_values)
        normalised_fitness = self.normaliser(fitness_values, pareto_ranks)

        for i in range(len(self.population_)):
            self.population_[i].internal_fitness_ = normalised_fitness[i]
        reference_points = das_dennis_points(self.n_reference_points, 2)

        for _ in range(self.n_iter):
            fitness_values = np.array([solution.internal_fitness_ for solution in self.population_])
            pareto_ranks = fast_non_dominated_sort(fitness_values)
            ref_direction_distance, closest_ref_direction = calc_ref_direction_distances(fitness_values,
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
            union_pop = self.population_ + mutated_children
            union_fitness_values = np.array([solution.fitness_ for solution in union_pop])
            union_pareto_ranks = fast_non_dominated_sort(union_fitness_values)
            # I think we need to normalise the population in every iteration outside of the following if clause
            # As the parent selection depends on objective values in U-NSGA-III
            union_fitness_values = self.normaliser(union_fitness_values, union_pareto_ranks)
            for l in range(len(union_pop)):
                union_pop[l].internal_fitness_ = union_fitness_values[l]
            union_pop = np.array(union_pop)
            tmp_pop = []
            l = 0
            while len(tmp_pop) + np.sum(union_pareto_ranks == l) < self.population_size:
                tmp_pop += union_pop[union_pareto_ranks == l].tolist()
                l += 1
            next_pop = tmp_pop.copy()
            tmp_pop += union_pop[union_pareto_ranks == l].tolist()

            if len(tmp_pop) == self.population_size:
                self.population_ = tmp_pop
                continue

            solutions_left_count = self.population_size - len(next_pop)

            union_ref_dist, union_closest_ref = calc_ref_direction_distances(union_fitness_values,
                                                                  reference_points)
            niche_count = np.zeros(reference_points.shape[0])
            values, counts = np.unique(union_closest_ref[union_pareto_ranks < l], return_counts=True)
            niche_count[values] = counts

            front_l = union_pop[union_pareto_ranks == l]
            front_l_closest_ref = union_closest_ref[union_pareto_ranks == l]
            front_l_ref_dist = union_ref_dist[union_pareto_ranks == l]
            # Niching
            k = 1
            while k < solutions_left_count:
                min_index = self.random_state_.choice(np.nonzero(niche_count == niche_count.min())[0])
                candidates = front_l[front_l_closest_ref == min_index]
                if len(candidates) == 0:
                    niche_count[min_index] = np.inf
                    continue
                if niche_count[min_index] == 0:
                    candidate_distances = front_l_ref_dist[front_l_closest_ref == min_index]
                    next_pop.append(candidates[np.argmin(candidate_distances)])
                else:
                    next_pop.append(np.random.choice(candidates))
                niche_count[min_index] = niche_count[min_index] + 1
                front_l_closest_ref = front_l_closest_ref[front_l != next_pop[-1]]
                front_l_ref_dist = front_l_ref_dist[front_l != next_pop[-1]]
                front_l = front_l[front_l != next_pop[-1]]
                k += 1

            self.population_ = next_pop
        return

    def pareto_front(self) -> list[Solution]:
        if not hasattr(self, "population_") or not self.population_:
            return []
        fitness_values = np.array([solution.fitness_ for solution in self.population_])
        pareto_ranks = fast_non_dominated_sort(fitness_values)
        pareto_front = np.array(self.population_)[pareto_ranks == 0]
        return sorted(pareto_front, key=lambda x: x.fitness_[0], reverse=True)

