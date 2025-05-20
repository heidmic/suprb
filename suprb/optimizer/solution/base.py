from abc import ABCMeta, abstractmethod
from typing import Union, Optional

import numpy as np

from suprb.solution import Solution, SolutionInit
from suprb.optimizer import BaseOptimizer
from suprb.rule import Rule
from suprb.utils import check_random_state
from .archive import SolutionArchive
from .sampler import SolutionSampler


class SolutionComposition(BaseOptimizer, metaclass=ABCMeta):
    """Base class of optimizers for `Solution`s.

    Parameters
    ----------
    n_iter: int
        Iterations the metaheuristic will perform.
    init: SolutionInit
    archive: SolutionArchive
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
    warm_start: bool
        If False, solutions are generated new for every `optimize()` call.
        If True, solutions are used from previous runs.
    n_jobs: int
        The number of threads / processes the optimization uses.
    """

    pool_: list[Rule]

    def __init__(
        self,
        n_iter: int,
        init: SolutionInit,
        archive: SolutionArchive,
        random_state: int,
        n_jobs: int,
        warm_start: bool,
    ):
        super().__init__(random_state=random_state, n_jobs=n_jobs)
        self.n_iter = n_iter
        self.init = init
        self.archive = archive
        self.warm_start = warm_start

    @abstractmethod
    def optimize(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Union[Solution, list[Solution], None]:
        pass

    @abstractmethod
    def elitist(self) -> Optional[Solution]:
        pass

    def _reset(self):
        super()._reset()
        if hasattr(self, "pool_"):
            del self.pool_


class PopulationBasedSolutionComposition(SolutionComposition, metaclass=ABCMeta):
    """Base class of population-based optimizers for `Solution`s."""

    pool_: list[Rule]
    population_: list[Solution]

    def __init__(
        self,
        population_size: int,
        n_iter: int,
        init: SolutionInit,
        archive: SolutionArchive,
        random_state: int,
        n_jobs: int,
        warm_start: bool,
    ):
        super().__init__(
            n_iter=n_iter,
            init=init,
            archive=archive,
            warm_start=warm_start,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.population_size = population_size

    def optimize(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Union[Solution, list[Solution], None]:
        """
        This method does all the work that is independent of the metaheuristic used.
        Specific procedures go into `_optimize()`.
        """

        self.random_state_ = check_random_state(self.random_state)

        self._init_population()

        # Pad solutions in the archive with zeros and update their fitness value,
        # because rules may were added to the pool
        if self.archive is not None:
            self.archive.pad()
            self.archive.pool_ = self.pool_
            self.archive.refit(X, y)

        if self.pool_:
            self._optimize(X, y)
        else:
            self.population_ = [self.init(self.pool_, self.random_state_).fit(X, y)]

        # Check if new solutions should be stored in the archive, store them and refit
        if self.archive is not None:
            self.archive(self.population_)
            self.archive.refit(X, y)

        return self.population_

    @abstractmethod
    def _optimize(self, X: np.ndarray, y: np.ndarray):
        """Method to overwrite in subclasses to perform the specific optimization."""
        pass

    def elitist(self) -> Optional[Solution]:
        """Returns the best `Solution` from the archive and the population together."""

        if not hasattr(self, "population_") or not self.population_:
            return None

        population = self.population_.copy()  # shallow copy
        if self.archive is not None:
            population.extend(self.archive.population_)
        return max(population, key=lambda elitist: elitist.fitness_)

    def _init_population(self):
        """Either generate new solutions or pad the existing ones using the initialization method."""

        if not self.warm_start or not hasattr(self, "population_") or not self.population_:
            self.population_ = [self.init(self.pool_, self.random_state_) for _ in range(self.population_size)]
        else:
            self.population_ = [self.init.pad(solution, self.random_state_) for solution in self.population_]

    def fit_population(self, X, y):
        self.population_ = [solution.fit(X, y) for solution in self.population_]

    def _reset(self):
        super()._reset()
        if hasattr(self, "population_"):
            del self.population_


def hypervolume(pareto_front: list[Solution]):
    pareto_front = sorted(pareto_front, key=lambda solution: solution.fitness_[0], reverse=True)
    fitness_values = np.array([solution.fitness_ for solution in pareto_front])
    # Needs a MultiObjectiveSolutionFitness
    reference_point = pareto_front[0].fitness.hv_reference_
    last_x = reference_point[0]
    volume = 0
    for fitness in fitness_values:
        volume += (last_x - fitness[0]) * np.prod(reference_point[1:] - fitness[1:])
        last_x = fitness[0]
    return volume


class MOSolutionComposition(PopulationBasedSolutionComposition, metaclass=ABCMeta):
    def __init__(
        self,
        population_size: int,
        n_iter: int,
        init: SolutionInit,
        archive: SolutionArchive,
        sampler: SolutionSampler,
        random_state: int,
        n_jobs: int,
        warm_start: bool,
        early_stopping_patience: int = -1,
        early_stopping_delta: float = 0,
    ):
        super().__init__(
            population_size=population_size,
            n_iter=n_iter,
            init=init,
            archive=archive,
            random_state=random_state,
            n_jobs=n_jobs,
            warm_start=warm_start,
        )
        self.sampler = sampler
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self._best_hypervolume = 0
        self._best_pareto_front = None
        self._early_stopping_counter = 0
        self.step_ = 0

    def check_early_stopping(self):
        self.step_ += 1
        if self.early_stopping_patience < 0:
            return False
        hv = self.hypervolume()
        hv_diff = hv - self._best_hypervolume
        if self._best_hypervolume < hv:
            self._best_hypervolume = hv
            self._best_pareto_front = self.pareto_front()

        if hv_diff > self.early_stopping_delta:
            self._early_stopping_counter = 0
        else:
            self._early_stopping_counter += 1
            if self.early_stopping_patience <= self._early_stopping_counter:
                print(
                    f"Execution was stopped early after {self.early_stopping_patience} cycles with no significant changes."
                )
                print(f"The early stopping criterion value was: {self.hypervolume()} after {self.step_} iterations.")
                return True
        return False

    @abstractmethod
    def _pareto_front(self) -> list[Solution]:
        pass

    def hypervolume(self) -> float:
        return hypervolume(self.pareto_front())

    def elitist(self) -> Optional[Solution]:
        """Sample an elitist from the Pareto front"""
        pf = self._pareto_front()
        if len(pf) == 0:
            return None
        return self.sampler(pf, random_state=self.random_state_)

    def optimize(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Union[Solution, list[Solution], None]:
        self._best_hypervolume = 0
        self._best_pareto_front = None
        self._early_stopping_counter = 0
        self.step_ = 0
        super().optimize(X, y, **kwargs)
