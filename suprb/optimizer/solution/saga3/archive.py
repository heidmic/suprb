from abc import abstractmethod, ABCMeta

import numpy as np

from suprb.base import BaseComponent
from suprb.optimizer.solution.archive import SolutionArchive
from suprb.optimizer.solution.saga3.solution_extension import SagaSolution
from suprb.solution import Solution
from suprb.rule import Rule


class SagaElitist(SolutionArchive):

    def __call__(self, new_population: list[SagaSolution]):
        best = max(new_population, key=lambda i: i.fitness_)
        if self.population_:
            if self.population_[0].fitness_ < best.fitness_:
                self.population_.pop(0)
                self.population_.append(best.clone())
        else:
            self.population_.append(best.clone())
