from abc import abstractmethod, ABCMeta

import numpy as np

from suprb2.base import BaseComponent
from suprb2.individual import Individual
from suprb2.rule import Rule


class IndividualArchive(BaseComponent, metaclass=ABCMeta):
    """Saves non-dominated `RulePopulation`s."""

    pool_: list[Rule]

    population_: list[Individual]

    def __init__(self):
        self.population_ = []

    def refit(self, X: np.ndarray, y: np.ndarray):
        self.population_ = [individual.fit(X, y) for individual in self.population_]

    def pad(self):
        for individual in self.population_:
            individual.genome = np.pad(individual.genome, (0, len(self.pool_) - individual.genome.shape[0]),
                                       mode='constant')

    @abstractmethod
    def __call__(self, new_population: list[Individual]):
        pass


class Elitist(IndividualArchive):

    def __call__(self, new_population: list[Individual]):
        best = max(new_population, key=lambda i: i.fitness_)
        if self.population_:
            if self.population_[0].fitness_ < best.fitness_:
                self.population_.pop(0)
                self.population_.append(best.clone())
        else:
            self.population_.append(best.clone())
