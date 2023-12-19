from abc import abstractmethod, ABCMeta

import numpy as np

from suprb.base import BaseComponent
from suprb.optimizer.solution.archive import SolutionArchive
from suprb.optimizer.solution.sas.solution_extension import SasSolution
from suprb.solution import Solution
from suprb.rule import Rule


class SasElitist(SolutionArchive):

    def __call__(self, new_population: list[SasSolution]):
        best = max(new_population, key=lambda i: i.fitness_)
        if self.population_:
            if self.population_[0].fitness_ < best.fitness_:
                self.population_.pop(0)
                self.population_.append(best.clone())
        else:
            self.population_.append(best.clone())
