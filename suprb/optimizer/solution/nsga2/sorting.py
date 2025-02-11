import numpy as np


def fast_non_dominated_sort(fitness_values: np.ndarray) -> np.ndarray:
    solution_count, objective_count = fitness_values.shape
    pareto_ranks = np.ones(solution_count) * -1
    dominated_count = np.zeros(solution_count)
    dominates_indices = [[] for _ in range(solution_count)]

    # Dominance Counts
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


if __name__ == "__main__":
    fitness_values = np.random.rand(29, 2)
    ranks = fast_non_dominating_sort(fitness_values)
    print(ranks)