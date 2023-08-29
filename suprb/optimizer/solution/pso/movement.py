from abc import ABCMeta

import numpy as np
from suprb import Solution
from suprb.base import BaseComponent
from suprb.utils import RandomState

from suprb.optimizer.solution.utils import sigmoid_binarize


class Particle(metaclass=ABCMeta):
    """Represents an abstract particle in PSO."""

    solution: Solution
    best_solution: Solution

    def update_best(self):
        pass

    def __repr__(self):
        return f"<{self.solution},best:{self.best_solution}>"


class ClassicalParticle(Particle):
    """Represents a classical particle with both position and velocity."""

    def __init__(self, velocity: np.ndarray, position: np.ndarray, solution: Solution):
        self.velocity = velocity
        self.position = position
        self.solution = solution

        self.best_position = position.copy()
        self.best_solution = solution.clone()

    def update_best(self):
        if self.solution.fitness_ > self.best_solution.fitness_:
            self.best_solution = self.solution.clone()
            self.best_position = self.position.copy()


class ParticleMovement(BaseComponent, metaclass=ABCMeta):
    """Moves the particles around in the search space somehow."""

    def init_particle(self, start: Solution, random_state: RandomState) -> Particle:
        pass

    def __call__(self, particles: list[Particle], a: float, random_state: RandomState):
        pass


class Sigmoid(ParticleMovement):
    """
    Performs classical PSO movement and binarizes the positions using sigmoid.

    Taken from https://doi.org/10/dhnq29.
    """

    def __init__(self, b: float = 1, c: float = 1, v_max: float = 1):
        self.b = b
        self.c = c
        self.v_max = v_max

    def init_particle(self, start: Solution, random_state: RandomState) -> Particle:
        position = (start.genome.astype(np.float) + random_state.random(size=start.genome.shape[0])) / 2
        velocity = random_state.uniform(-self.v_max, self.v_max, size=start.genome.shape[0])

        return ClassicalParticle(velocity=velocity, position=position, solution=start)

    def __call__(self, particles: list[ClassicalParticle], a: float, random_state: RandomState):
        global_best_position = max(particles, key=lambda p: p.solution.fitness_).position
        n = particles[0].position.shape[0]

        for particle in particles:
            br = self.b * random_state.random(size=n) * (particle.best_position - particle.position)
            cr = self.c * random_state.random(size=n) * (global_best_position - particle.position)
            particle.velocity = np.clip(a * particle.velocity + br + cr, -self.v_max, self.v_max)
            particle.position = np.clip(particle.position + particle.velocity, 0, 1)
            particle.solution.genome = sigmoid_binarize(particle.position, random_state=random_state)


class QuantumParticle(Particle):
    """A 'quantum' particle with only position, and no velocity."""

    def __init__(self, position: np.ndarray, solution: Solution):
        self.position = position
        self.solution = solution

        self.best_position = position.copy()
        self.best_solution = solution.clone()

    def update_best(self):
        if self.solution.fitness_ > self.best_solution.fitness_:
            self.best_solution = self.solution.clone()
            self.best_position = self.position.copy()


class SigmoidQuantum(ParticleMovement):
    """
    Performs Quantum-PSO movement and binarizes the positions using sigmoid.

    Taken from https://doi.org/10/b6dd5w.
    """

    def init_particle(self, start: Solution, random_state: RandomState) -> Particle:
        position = (start.genome.astype(np.float) + random_state.random(size=start.genome.shape[0])) / 2

        return QuantumParticle(position=position, solution=start)

    def __call__(self, particles: list[QuantumParticle], a: float, random_state: RandomState):
        global_best_position = max(particles, key=lambda p: p.solution.fitness_).position
        mean_position = np.mean([particle.position for particle in particles], axis=0)
        n = particles[0].position.shape[0]

        for particle in particles:
            lmbda = random_state.random(size=n) <= 0.5
            mixed_global = lmbda * particle.position + (1 - lmbda) * global_best_position
            mixed_mean = a * np.abs(mean_position - particle.position) * np.log(1 / random_state.random(size=n))
            particle.position = np.clip(mixed_global + mixed_mean * random_state.choice([-1, 1]), 0, 1)
            particle.solution.genome = sigmoid_binarize(particle.position, random_state=random_state)


class BinaryParticle(Particle):
    """Represents a particle completely in binary space."""

    def __init__(self, solution: Solution):
        self.solution = solution
        self.best_solution = solution.clone()

    def update_best(self):
        if self.solution.fitness_ > self.best_solution.fitness_:
            self.best_solution = self.solution.clone()


def binary_mean(x: list[np.ndarray], random_state: RandomState) -> np.ndarray:
    """
    Calculate the mean of x in binary, e.g. a 1 is at index i if more particles have a 1 at position i
    than there are particles with 0 at index i and vice versa.
    """

    x_sum = np.sum(x, axis=0)
    half = x_sum.shape[0] // 2
    equal_indices = x_sum == half
    mean = (x_sum >= half)
    mean[equal_indices] = random_state.integers(0, 2, size=np.count_nonzero(equal_indices))

    return mean.astype(np.bool)


def best_random_subset_particle(particles: list[Particle], n: int, random_state: RandomState) -> Particle:
    """Calculates the best particle of a random subset of particles with size n."""

    subset = random_state.choice(particles, size=n)
    return max(subset, key=lambda p: p.best_solution.fitness_)


def hamming_distance(x1: np.ndarray, x2: np.ndarray) -> int:
    """Calculates the hamming distance between two binary arrays."""

    return np.count_nonzero(x1 ^ x2)


class BinaryQuantum(ParticleMovement):
    """
    Performs 'quantum' movement of particles completely in binary space.

    Taken from https://doi.org/10/gnxcd9.

    The original versions additionally considers mutation and crossover of particles, but they are omitted here.
    """

    def __init__(self, p_learning: float = 0.5, n_attractors: int = 2):
        self.p_learning = p_learning
        self.n_attractors = n_attractors

    def init_particle(self, start: Solution, random_state: RandomState) -> Particle:
        return BinaryParticle(solution=start)

    def __call__(self, particles: list[BinaryParticle], a: float, random_state: RandomState):
        mean_genome = binary_mean([particle.solution.genome for particle in particles], random_state=random_state)
        n = particles[0].solution.genome.shape[0]

        # Calculate attractors for every particle
        attractors = []
        for particle in particles:
            genome = particle.best_solution.genome.copy()
            indices = np.nonzero(random_state.random(size=n) <= self.p_learning)

            # Get some indices from the best particle of a random subset
            attractor = []
            for i in indices:
                neighbor = best_random_subset_particle(particles, n=self.n_attractors, random_state=random_state)
                attractor.append(neighbor.best_solution.genome[i])

            genome[indices] = attractor

            # If no index is different from the own best, change a random dimension
            if np.all(genome == particle.best_solution.genome):
                i = random_state.integers(0, n)
                neighbor = best_random_subset_particle(particles, n=self.n_attractors, random_state=random_state)
                genome[i] = neighbor.best_solution.genome[i]

            attractors.append(genome)

        for particle, attractor in zip(particles, attractors):
            # Calculate the probability to use the attractor
            b = a * hamming_distance(particle.solution.genome, mean_genome) * np.log(1 / random_state.random())
            pr = min(b / n, 1)

            # Mix the old position and the attractor
            mask = random_state.random(size=n) <= pr
            particle.solution.genome[mask] = attractor[mask]
