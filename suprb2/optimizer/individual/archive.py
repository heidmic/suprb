from abc import abstractmethod

import numpy as np

from suprb2.base import BaseComponent
from suprb2.individual import Individual
from suprb2.optimizer.individual.fitness import IndividualFitness
from suprb2.rule import Rule


class IndividualArchive(BaseComponent):
    """Saves non-dominated `RulePopulation`s."""

    pool_: list[Rule]
    population_: list[Individual]

    @abstractmethod
    def refit(self, X: np.ndarray, y: np.ndarray, fitness: IndividualFitness):
        pass

    @abstractmethod
    def pad(self):
        pass


class Elitist(IndividualArchive):

    def refit(self, X: np.ndarray, y: np.ndarray, fitness: IndividualFitness):
        pass

    def pad(self):
        pass
