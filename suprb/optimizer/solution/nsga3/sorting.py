from typing import Tuple

import numpy as np


def fast_non_dominated_sort(fitness_values: np.ndarray) -> np.ndarray:
    """Sorts the fitness values into multiple levels of non domination.

    Parameters
    ----------
    fitness_values: np.ndarray
        numpy of shape (solution_count, objective_count) with all fitness values for every solution

    Returns
    -------
    pareto ranks: level of non-dominated front for each solution (non sorted)

    """
    solution_count, objective_count = fitness_values.shape
    pareto_ranks = np.ones(solution_count, dtype=np.int32) * -1
    dominated_count = np.zeros(solution_count, dtype=np.int32)
    dominates_indices = [[] for _ in range(solution_count)]

    # Calculating the amount of solutions, that dominate i and
    # the solutions i is dominating
    for i in range(solution_count):
        fitness_i = fitness_values[i][None, :]

        less_equal_count = np.sum(fitness_values <= fitness_i, axis=1)
        less_than_count = np.sum(fitness_values < fitness_i, axis=1)
        dominates_i_mask = (less_equal_count == objective_count) & (less_than_count >= 1)
        dominated_count[i] = np.sum(dominates_i_mask)

        greater_equal_count = np.sum(fitness_values >= fitness_i, axis=1)
        greater_than_count = np.sum(fitness_values > fitness_i, axis=1)
        dominated_by_i_mask = (greater_equal_count == objective_count) & (greater_than_count >= 1)
        dominates_indices[i] = np.argwhere(dominated_by_i_mask).flatten().tolist()

    pareto_ranks[dominated_count == 0] = 0
    front_rank = 0

    # Actual non dominated sort
    while -1 in pareto_ranks:
        current_front = np.argwhere(pareto_ranks == front_rank).flatten()
        front_rank += 1
        for solution in current_front:
            solution_mask = np.zeros(solution_count, dtype=bool)
            solution_mask[dominates_indices[solution]] = 1
            solution_mask = solution_mask & (dominated_count == 1)
            dominated_count[dominates_indices[solution]] -= 1
            pareto_ranks[solution_mask] = front_rank

    return pareto_ranks


def calculate_crowding_distances(fitness_values: np.ndarray, pareto_ranks: np.ndarray) -> np.ndarray:
    solution_count, objective_count = fitness_values.shape
    crowding_distances = np.zeros(solution_count)

    for rank in np.unique(pareto_ranks):
        front_indices = np.where(pareto_ranks == rank)[0]
        front_size = len(front_indices)

        if front_size <= 2:
            crowding_distances[front_indices] = np.inf
            continue

        front_fitness = fitness_values[front_indices]

        for m in range(objective_count):
            sorting_permutation = np.argsort(front_fitness[:, m])
            sorted_front = front_indices[sorting_permutation]

            crowding_distances[sorted_front[0]] = np.inf
            crowding_distances[sorted_front[-1]] = np.inf

            min_f = front_fitness[sorting_permutation[0], m]
            max_f = front_fitness[sorting_permutation[-1], m]

            # if max_f == min_f the crowding distance parts that result from objective_m are all 0 as all
            # solution share the same coordinate in this dimension of the fitness function
            if max_f > min_f:
                normalized_range = max_f - min_f
                for i in range(1, front_size - 1):
                    crowding_distances[sorted_front[i]] += (
                        front_fitness[sorting_permutation[i + 1], m] - front_fitness[sorting_permutation[i - 1], m]
                    ) / normalized_range

    return crowding_distances
