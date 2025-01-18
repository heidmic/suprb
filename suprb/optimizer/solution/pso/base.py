import numpy as np

from suprb.solution import SolutionInit
from suprb.solution.initialization import RandomInit
from .movement import ParticleMovement, Sigmoid, Particle
from ..archive import Elitist, SolutionArchive
from ..base import PopulationBasedSolutionComposition


class ParticleSwarmOptimization(PopulationBasedSolutionComposition):
    """Particle Swarm Optimization written in Python.

    The base version was taken from https://doi.org/10/bdc3t3.

    Parameters
    ----------
    n_iter: int
        Iterations the metaheuristic will perform.
    population_size: int
        Number of solutions in the population.
    a_min: float
        Inertia weight at the last iteration.
    a_max: float
        Inertia weight at the first iteration.
    movement: ParticleMovement
    init: SolutionInit
    archive: Archive
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
    warm_start: bool
        If False, solutions are generated new for every `optimize()` call.
        If True, solutions are used from previous runs.
    n_jobs: int
        The number of threads / processes the optimization uses.
    """

    particles: list[Particle]

    def __init__(
        self,
        n_iter: int = 32,
        population_size: int = 32,
        a_min: float = 1,
        a_max: float = 2,
        movement: ParticleMovement = Sigmoid(),
        init: SolutionInit = RandomInit(),
        archive: SolutionArchive = Elitist(),
        random_state: int = None,
        n_jobs: int = 1,
        warm_start: bool = True,
    ):
        super().__init__(
            n_iter=n_iter,
            population_size=population_size,
            init=init,
            archive=archive,
            random_state=random_state,
            n_jobs=n_jobs,
            warm_start=warm_start,
        )

        self.a_min = a_min
        self.a_max = a_max

        self.movement = movement

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        self.fit_population(X, y)

        a = self.a_max
        step_size = (self.a_max - self.a_min) / self.n_iter

        # Initialize particles
        self.particles = [
            self.movement.init_particle(start=solution, random_state=self.random_state_)
            for solution in self.population_
        ]

        for _ in range(self.n_iter):
            # Perform movement of particles
            self.movement(
                particles=self.particles, a=a, random_state=self.random_state_
            )

            a -= step_size

            # Refit and update their local best
            self.fit_population(X, y)

            for particle in self.particles:
                particle.update_best()
