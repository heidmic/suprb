from suprb.optimizer.solution.archive import SolutionArchive
from suprb.solution import Solution
from .internal_fitness import calculate_raw_internal_fitness, calculate_density, distance_to_kth
import numpy as np


class EnvironmentalArchive(SolutionArchive):

    def __init__(self, max_population_size, kth_nearest):
        super().__init__()
        self.max_population_size = max_population_size
        self.kth_nearest = kth_nearest

    def __call__(self, new_population: list[Solution]):
        pop_size = len(new_population)
        fitness_values = np.array([solution.fitness_ for solution in new_population + self.population_])
        pop_and_arch = np.array([solution for solution in new_population + self.population_])
        raw_internal_fitness_values = calculate_raw_internal_fitness(fitness_values)
        density_values = calculate_density(fitness_values, self.kth_nearest)
        internal_fitness_values = raw_internal_fitness_values + density_values

        # Pass the internal fitness values to the optimizer to avoid recalculation ...?
        for i, solution in enumerate(pop_and_arch):
            solution.internal_fitness_ = internal_fitness_values[i]

        sorting_permutation = np.argsort(internal_fitness_values)
        internal_fitness_values = internal_fitness_values[sorting_permutation]
        pop_and_arch = pop_and_arch[sorting_permutation].tolist()
        # We can index the 0 directly as there is always a pareto dominant element in a finite set.
        non_dominated_count = np.unique(internal_fitness_values, return_counts=True)[1][0]

        if non_dominated_count < self.max_population_size:
            self.population_ = pop_and_arch[: self.max_population_size]
        else:
            # In this case the population needs to be truncated!
            self.population_ = pop_and_arch[:non_dominated_count]
            # TODO: Vectorize this to achieve better performance
            while len(self.population_) > self.max_population_size:
                archive_fitness_values = np.array([solution.fitness_ for solution in self.population_])
                distances = np.zeros(len(self.population_))
                candidates_mask = np.ones(len(distances), dtype=np.bool8)
                for k in range(1, len(self.population_)):
                    for i in range(len(self.population_)):
                        distances[i] = distance_to_kth(archive_fitness_values[i], archive_fitness_values, k)
                    unique = np.unique(distances[candidates_mask])[0]
                    candidates_mask = (distances == unique) & candidates_mask
                    if np.sum(candidates_mask) == 1 or k == len(self.population_) - 1:
                        index = np.argmax(candidates_mask)
                        self.population_.pop(index)
                        break
