from itertools import combinations_with_replacement
import numpy as np


def das_dennis_points(num_partitions: int, num_dimensions: int = 2) -> np.ndarray:
    """
    Generate evenly distributed points within a simplex using the Das-Dennis methodology.

    Parameters
    ----------
        num_partitions: int
            The number of partitions (grids) to divide the simplex.
        num_dimensions:
            int The number of dimensions for the simplex. Default is 2.

    Returns
    -------
        np.ndarray
            A 2D NumPy array where each row represents a point within the simplex.
    """

    partitions = combinations_with_replacement(range(num_dimensions), num_partitions)
    points = []
    for part in partitions:
        coord = np.zeros(num_dimensions)
        for idx in part:
            coord[idx] += 1
        points.append(coord / num_partitions)
    return np.array(points)


def calc_ref_direction_distances(fitness_values: np.ndarray, reference_points: np.ndarray) -> tuple[
    np.ndarray, np.ndarray]:
    """
    Calculate distances between fitness values and reference points, and find the closest reference points.

    Parameters
    ----------
        fitness_values : np.ndarray
            A 2D array where each row represents a solution's fitness values.
        reference_points : np.ndarray
            A 2D array where each row represents a predefined reference direction.

    Returns
    -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - min_distances: A 1D array representing the minimum distance from each solution to the closest reference point.
            - min_ref_points: A 1D array containing the index of the closest reference point for each solution.
    """
    solution_count, _ = fitness_values.shape
    reference_count, _ = reference_points.shape
    min_distances = np.zeros(solution_count)
    min_ref_points = np.zeros(solution_count)

    for fit_idx in range(solution_count):
        distances = np.zeros(reference_count)
        for ref_idx in range(reference_points.shape[0]):
            numerator = np.dot(np.dot(reference_points[ref_idx].T, fitness_values[fit_idx]), reference_points[ref_idx])
            denominator = np.linalg.norm(reference_points) ** 2
            distances[ref_idx] = np.linalg.norm(fitness_values[fit_idx] - (numerator / denominator))
        min_distances[fit_idx] = np.min(distances)
        min_ref_points[fit_idx] = np.argmin(distances)
    return min_distances, min_ref_points


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    points = das_dennis_points(6, 2)
    if points.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.plot(points[:,0], points[:,1], points[:,2], 'o')
        ax.view_init(elev=10, azim=-80)
    else:
        plt.plot(points[:,0], points[:,1], 'o')
    plt.show()
    print(len(points))