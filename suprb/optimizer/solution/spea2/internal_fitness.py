import numpy as np


def calculate_raw_internal_fitness(fitness_values: np.ndarray) -> np.ndarray:
    """Calculates the raw internal fitness values R(i) for each solution.
    These are the added strength values of all dominators of i.

    Parameters
    ----------
    fitness_values: np.ndarray
        numpy array of shape (solution_count, objective_count) with all fitness values for every solution

    Returns
    -------
    raw_internal_fitness_values: np.ndarray
       1D numpy array of length solution_count with the raw internal fitness values for every solution
    """
    solution_count, objective_count = fitness_values.shape
    strength_values = np.zeros(solution_count, dtype=np.int32)
    dominator_indices = [[]] * solution_count
    raw_internal_fitness_values = np.zeros(solution_count)

    # First we calculate the strength values S(i) and the indices of Solutions that dominate i.
    for i in range(solution_count):
        fitness_i = fitness_values[i][None, :]
        greater_equal_count = np.sum(fitness_values >= fitness_i, axis=1)
        greater_than_count = np.sum(fitness_values > fitness_i, axis=1)
        dominated_by_i_mask = (greater_equal_count == objective_count) & (greater_than_count >= 1)
        strength_values[i] = np.sum(dominated_by_i_mask)

        less_equal_count = np.sum(fitness_values <= fitness_i, axis=1)
        less_than_count = np.sum(fitness_values < fitness_i, axis=1)
        dominates_i_mask = (less_equal_count == objective_count) & (less_than_count >= 1)
        dominator_indices[i] = np.argwhere(dominates_i_mask).flatten().tolist()

    # The raw internal fitness values are calculated by summing up the strength values of
    # all dominators of one solution.

    for i in range(solution_count):
        if len(dominator_indices[i]) > 0:
            raw_internal_fitness_values[i] = np.sum(strength_values[np.array(dominator_indices[i])])

    return raw_internal_fitness_values


def calculate_density(fitness_values: np.ndarray, k: int) -> np.ndarray:
    """Calculates the density D(i) for each solution.

    Parameters
    ----------
    fitness_values: np.ndarray
        numpy array of shape (solution_count, objective_count) with all fitness values for every solution
    k: int
        The distance to the k-th nearest neighbour is selected

    Returns
    -------
    density_values: np.ndarray
           1D numpy array of length solution_count with the density values for every solution
    """
    solution_count, objective_count = fitness_values.shape
    density_values = np.zeros(solution_count, dtype=np.int32)

    for i in range(solution_count):
        density_values[i] = distance_to_kth(fitness_values[i], fitness_values, k)

    return density_values


def distance_to_kth(fitness_i:np.ndarray, fitness_values: np.ndarray, k: int) -> np.ndarray:
    distances = np.linalg.norm(fitness_i[None, :] - fitness_values, axis=-1)
    distances = np.sort(distances)
    return distances[k]

