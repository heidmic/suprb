from abc import ABCMeta

import numpy as np
from suprb import Solution
from suprb.base import BaseComponent
from suprb.utils import RandomState

from suprb.optimizer.solution.utils import sigmoid_binarize


class SolutionPositionUpdate(BaseComponent, metaclass=ABCMeta):
    """Calculates the next positions of all wolves, using the leaders."""

    def __call__(
        self,
        leaders: list[Solution],
        population: list[Solution],
        a: float,
        random_state: RandomState,
    ) -> list[Solution]:
        pass


def binary_C(prey: np.ndarray, wolf: np.ndarray, a: float, random_state: RandomState) -> np.ndarray:
    a_pos = A(a=a, n=prey.shape[0], random_state=random_state)
    d_pos = D(prey=prey, wolf=wolf, random_state=random_state)

    return sigmoid_binarize(a_pos * d_pos, random_state=random_state)


def binary_B(prey: np.ndarray, wolf: np.ndarray, a: float, random_state: RandomState) -> np.ndarray:
    c_pos = binary_C(prey=prey, wolf=wolf, a=a, random_state=random_state)

    return c_pos >= random_state.random(size=prey.shape[0])


def binary_X(leader: np.ndarray, wolf: np.ndarray, a: float, random_state: RandomState) -> np.ndarray:
    leader = leader.astype(np.float)
    wolf = wolf.astype(np.float)

    b_pos = binary_B(prey=leader, wolf=wolf, a=a, random_state=random_state)

    return (leader + b_pos) >= 1


def A(a: float, n: int, random_state: RandomState) -> np.ndarray:
    return 2 * a * random_state.random(size=n) - a


def D(prey: np.ndarray, wolf: np.ndarray, random_state: RandomState) -> np.ndarray:
    return np.abs(2 * random_state.random(size=prey.shape[0]) * prey - wolf)


def X(leader: np.ndarray, wolf: np.ndarray, a: float, random_state: RandomState) -> np.ndarray:
    leader = leader.astype(np.float)
    wolf = wolf.astype(np.float)

    a_pos = A(a=a, n=leader.shape[0], random_state=random_state)
    d_pos = D(prey=leader, wolf=wolf, random_state=random_state)

    return np.abs(leader - a_pos * d_pos)


class Crossover(SolutionPositionUpdate):
    """
    Performs a stochastic crossover to update the positions.

    Taken from https://doi.org/10/gfxvkw.
    """

    def __call__(
        self,
        leaders: list[Solution],
        population: list[Solution],
        a: float,
        random_state: RandomState,
    ) -> list[Solution]:
        new_population = []
        for solution in population:
            # Calculate binary vectors for solution with each leader
            xs = [
                binary_X(
                    leader=leader.genome,
                    wolf=solution.genome,
                    a=a,
                    random_state=random_state,
                )
                for leader in leaders
            ]
            xs = np.stack(xs, axis=0)

            # Uniform crossover
            crossover = np.choose(random_state.integers(len(leaders), size=solution.genome.shape[0]), xs)

            new_population.append(solution.clone(genome=crossover))

        return new_population


class Sigmoid(SolutionPositionUpdate):
    """
    Binarizes the continuous positions using stochastic sigmoid.

    Taken from https://doi.org/10/gfxvkw.
    """

    def __call__(
        self,
        leaders: list[Solution],
        population: list[Solution],
        a: float,
        random_state: RandomState,
    ) -> list[Solution]:
        new_population = []
        for solution in population:
            # Calculate binary vectors for solution with each leader
            xs = [
                X(
                    leader=leader.genome,
                    wolf=solution.genome,
                    a=a,
                    random_state=random_state,
                )
                for leader in leaders
            ]
            xs = np.stack(xs, axis=0)

            # Binarize the vectors through sigmoid
            mean = xs.mean(axis=0)
            binarized = sigmoid_binarize(mean, random_state=random_state)

            new_population.append(solution.clone(genome=binarized))

        return new_population
