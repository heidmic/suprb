from abc import ABCMeta
from typing import Union

import numpy as np

from suprb2.individual import Individual, IndividualInit
from suprb2.optimizer import BaseOptimizer
from suprb2.rule import Rule
from suprb2.utils import check_random_state
from . import IndividualArchive


class IndividualOptimizer(BaseOptimizer, metaclass=ABCMeta):
    """ Base class of optimizers for `Individual`s.

    Parameters
    ----------
    n_iter: int
        Iterations the the metaheuristic will perform.
    init: IndividualInit
    archive: IndividualArchive
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
    warm_start: bool
        If False, individuals are generated new for every `optimize()` call.
        If True, individuals are used from previous runs.
    n_jobs: int
        The number of threads / processes the optimization uses.
    """

    pool_: list[Rule]

    def __init__(self,
                 n_iter: int,
                 init: IndividualInit,
                 archive: IndividualArchive,
                 random_state: int,
                 n_jobs: int,
                 warm_start: bool,
                 ):
        super().__init__(random_state=random_state, n_jobs=n_jobs)
        self.n_iter = n_iter
        self.init = init
        self.archive = archive
        self.warm_start = warm_start

    def optimize(self, X: np.ndarray, y: np.ndarray) -> Union[Individual, list[Individual], None]:
        pass

    def _reset(self):
        super()._reset()
        if hasattr(self, 'pool_'):
            del self.pool_


class PopulationBasedIndividualOptimizer(IndividualOptimizer, metaclass=ABCMeta):
    """Base class of population-based optimizers for `Individual`s."""

    pool_: list[Rule]
    population_: list[Individual]

    def __init__(self,
                 population_size: int,
                 n_iter: int,
                 init: IndividualInit,
                 archive: IndividualArchive,
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
            n_jobs=n_jobs)
        self.population_size = population_size

    def optimize(self, X: np.ndarray, y: np.ndarray) -> Union[Individual, list[Individual], None]:
        """
        This method does all the work that is independent of the metaheuristic used.
        Specific procedures go into `_optimize()`.
        """

        self.random_state_ = check_random_state(self.random_state)

        self._init_population()

        if self.archive is not None:
            self.archive.pad()
            self.archive.pool_ = self.pool_

        if self.pool_:
            self._optimize(X, y)
        else:
            self.population_ = [self.init(self.pool_, self.random_state_)]

        if self.archive is not None:
            self.archive(self.population_)
            self.archive.refit(X, y)

        return self.population_

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        """Method to overwrite in subclasses to perform the specific optimization."""
        pass

    def elitist(self) -> Individual:
        """Returns the best `Individual` from the archive and the population together."""
        population = self.population_.copy()  # shallow copy
        if self.archive is not None:
            population.extend(self.archive.population_)
        return max(population, key=lambda elitist: elitist.fitness_)

    def _init_population(self):
        """Either generate new individuals or pad the existing ones using the initialization method."""

        if not self.warm_start or not hasattr(self, 'population_') or not self.population_:
            self.population_ = [self.init(self.pool_, self.random_state_) for _ in range(self.population_size)]
        else:
            self.population_ = [self.init.pad(individual, self.random_state_) for individual in self.population_]

    def fit_population(self, X, y):
        self.population_ = [individual.fit(X, y) for individual in self.population_]

    def _reset(self):
        super()._reset()
        if hasattr(self, 'population_'):
            del self.population_
