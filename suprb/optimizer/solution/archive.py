from abc import abstractmethod, ABCMeta

import numpy as np

from suprb.base import BaseComponent
from suprb.solution import Solution
from suprb.rule import Rule


class SolutionArchive(BaseComponent, metaclass=ABCMeta):
    """Saves non-dominated `RulePopulation`s."""

    pool_: list[Rule]

    population_: list[Solution]

    def __init__(self):
        self.population_ = []

    def refit(self, X: np.ndarray, y: np.ndarray):
        self.population_ = [solution.fit(X, y) for solution in self.population_]

    def pad(self):
        for solution in self.population_:
            solution.genome = np.pad(
                solution.genome,
                (0, len(self.pool_) - solution.genome.shape[0]),
                mode="constant",
            )

    @abstractmethod
    def __call__(self, new_population: list[Solution]):
        pass


class Elitist(SolutionArchive):

    def __call__(self, new_population: list[Solution]):
        best = max(new_population, key=lambda i: i.fitness_)
        if self.population_:
            if self.population_[0].fitness_ < best.fitness_:
                self.population_.pop(0)
                self.population_.append(best.clone())
        else:
            self.population_.append(best.clone())
