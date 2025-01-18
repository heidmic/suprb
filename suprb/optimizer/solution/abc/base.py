import numpy as np

from suprb import Solution
from suprb.solution import SolutionInit
from suprb.solution.initialization import RandomInit
from .food import FoodSourceUpdate, Sigmoid, FoodSource
from ..archive import Elitist, SolutionArchive
from ..base import PopulationBasedSolutionComposition


class ArtificialBeeColonyAlgorithm(PopulationBasedSolutionComposition):
    """Artificial Bee Colony Algorithm written in Python.

    The base version was taken from https://doi.org/10/c25dm6.

    Parameters
    ----------
    n_iter: int
        Iterations the metaheuristic will perform.
    population_size: int
        Number of solutions in the population.
    trials_limit: float
        Number of trials to enhance an solution before it is reset.
    food: FoodSourceUpdate
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

    food_sources_: list[FoodSource]

    def __init__(
        self,
        n_iter: int = 32,
        population_size: int = 32,
        trials_limit: int = 25,
        food: FoodSourceUpdate = Sigmoid(),
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

        self.trials_limit = trials_limit
        self.food = food

    def greedy_update(self, new_solutions: list[Solution]):
        """
        Update the food sources greedily. If no update is performed,
        the trails counter is increased, otherwise it is reset.
        """

        for new_solution, food_source in zip(new_solutions, self.food_sources_):
            if new_solution.fitness_ > food_source.solution.fitness_:
                food_source.solution = new_solution
                food_source.trials = 0
            else:
                food_source.trials += 1

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        self.fit_population(X, y)

        self.food_sources_ = [FoodSource(solution) for solution in self.population_]

        for _ in range(self.n_iter):

            # Employed bee phase: Crossover with random other solution
            new_solutions = []
            other_food_sources = self.random_state_.choice(self.food_sources_, size=self.population_size)
            for food_source, other in zip(self.food_sources_, other_food_sources):
                new = self.food(food_source, other, random_state=self.random_state_).fit(X, y)
                new_solutions.append(new)

            self.greedy_update(new_solutions)

            # Outlooker bee phase: Crossover with roulette wheel selection
            new_solutions = []
            weights = np.array([source.solution.fitness_ for source in self.food_sources_])
            normalized_weights = weights / weights.sum()
            other_food_sources = self.random_state_.choice(
                self.food_sources_, p=normalized_weights, size=self.population_size
            )
            for food_source, other in zip(self.food_sources_, other_food_sources):
                new = self.food(food_source, other, random_state=self.random_state_).fit(X, y)
                new_solutions.append(new)

            self.greedy_update(new_solutions)

            # Scout bee phase: Reinitialize the food sources with trails greater than the limit
            for food_source in self.food_sources_:
                if food_source.trials >= self.trials_limit:
                    food_source.solution = self.init(self.pool_, random_state=self.random_state_)
                    food_source.solution.fit(X, y)
                    food_source.trials = 0

            # Update the population
            self.population_ = [food_source.solution for food_source in self.food_sources_]
