import traceback
from abc import ABCMeta, abstractmethod

from suprb.solution import Solution

import numpy as np

from suprb.base import BaseComponent


class NSGAIIINormaliser(BaseComponent, metaclass=ABCMeta):

    def __init__(self) -> None:
        self._ideal_point = None
        self._nadir_point = None

    def _update_ideal_point(self, fitness_values: np.ndarray) -> np.ndarray:
        if self._ideal_point is not None:
            fitness_values = np.concatenate([fitness_values, self._ideal_point[np.newaxis, :]])

        self._ideal_point = np.min(fitness_values, axis=0)
        return self._ideal_point

    @abstractmethod
    def _update_nadir_point(self, fitness_values: np.ndarray, pareto_ranks: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, fitness_values: np.ndarray, pareto_ranks: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        fitness_values : np.ndarray
            Fitness values from which to update ideal and nadir points.
        pareto_ranks: np.ndarray
            Associated levels of pareto domination used to update nadir points.

        Returns
        -------
        np.ndarray
            Normalised fitness values.
        """
        self._update_ideal_point(fitness_values)
        self._update_nadir_point(fitness_values, pareto_ranks)
        return (fitness_values - self._ideal_point) / (self._nadir_point - self._ideal_point)


class HyperPlaneNormaliser(NSGAIIINormaliser):
    """
    Implemented as described in 10.1007/978-3-030-12598-1_19
    """

    def __init__(self, objective_count, epsilon_asf: float = 10e-6, epsilon_nad: float = 10e-6) -> None:
        super().__init__()
        self.objective_count = objective_count
        self.epsilon_asf = epsilon_asf
        self._extreme_points: np.ndarray | None = None
        self._worst_points_estimate = np.zeros(self.objective_count)
        self.epsilon_nad = epsilon_nad

    def _update_extreme_points(self, fitness_values: np.ndarray) -> np.ndarray:
        if self._extreme_points is not None:
            fitness_values = np.concatenate([fitness_values, self._extreme_points])
        else:
            self._extreme_points = np.zeros((self.objective_count, self.objective_count))

        for obj_idx in range(self.objective_count):
            weights = np.ones(self.objective_count) * self.epsilon_asf
            weights[obj_idx] = 1
            scalarised = asf(fitness_values, self._ideal_point, weights)
            k = np.argmin(scalarised)
            self._extreme_points[obj_idx] = fitness_values[k]
        return self._extreme_points

    def _update_nadir_point(self, fitness_values: np.ndarray, pareto_ranks: np.ndarray) -> np.ndarray:
        self._worst_points_estimate = np.max(
            np.concatenate([fitness_values, self._worst_points_estimate[None, :]]), axis=0
        )
        self._update_extreme_points(fitness_values)
        try:
            normal, distance = find_plane_from_points(self._extreme_points - self._ideal_point)
            intercepts = find_hyperplane_axis_intercepts(normal, distance)
            nadir_point = np.zeros(self.objective_count)
            for k in range(self.objective_count):
                nadir_k = intercepts[k] + self._ideal_point[k]
                if intercepts[k] < self.epsilon_nad or nadir_k > self._worst_points_estimate[k]:
                    raise ValueError("Nadir point outside worst point")
                nadir_point[k] = intercepts[k] + self._ideal_point[k]

        except Exception as e:
            print(traceback.format_exc())
            print("Falling back to per-axis maximum normalisation!")
            nadir_point = np.max(fitness_values, axis=0)

        fallback_nadir = np.max(fitness_values[pareto_ranks == 0], axis=0)
        mask = nadir_point - self._ideal_point < self.epsilon_nad
        nadir_point[mask] = fallback_nadir[mask]
        self._nadir_point = nadir_point
        return nadir_point


def find_hyperplane_axis_intercepts(normal_vector, distance) -> np.ndarray:
    """
    Compute intercepts of a hyperplane with coordinate axes.

    Parameters
    ----------
    normal_vector: np.ndarray
        Normal vector of the hyperplane [n1, n2, ..., nn].
    distance: float
        Distance of the hyperplane from the origin (d in equation).

    Returns
    -------
    intercepts: np.ndarray
        Intercepts with each axis as an n dimensional vector.
    """
    normal_vector = np.array(normal_vector)
    objective_count = normal_vector.shape[0]
    intercepts = np.zeros(objective_count)

    for k in range(objective_count):
        if normal_vector[k] != 0:
            # Set all other coordinates to zero and solve for this axis
            point = np.zeros_like(normal_vector)
            intercepts[k] = -distance / normal_vector[k]
        else:
            # No intersection if normal component is zero
            raise ValueError("Normal vector must not contain zero components.")
    return intercepts


def find_plane_from_points(points: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute a n dimensional hyperplane from a set of n points.

    Parameters
    ----------
    points: np.ndarray
        Set of points [n1, n2, ..., nn]. from which the Hyperplane is computed.

    Returns
    -------
    normal, distance: tuple[np.ndarray, float]
        The normal vector of the hyperplane as an n dimensional numpy array and the orthogonal distance.
    """
    center = np.mean(points, axis=0)
    centered_points = points - center
    _, _, vh = np.linalg.svd(centered_points)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    distance = -np.dot(normal, center)
    return normal, distance


def asf(fitness_values: np.ndarray, ideal_point: np.ndarray, weights: np.ndarray) -> float:
    return np.max((fitness_values - ideal_point) / weights, axis=1)
