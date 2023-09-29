from abc import ABCMeta

import numpy as np

from suprb.base import BaseComponent
from .solution_extension import SasSolution
from suprb.utils import RandomState


class SolutionSelection(BaseComponent, metaclass=ABCMeta):

    def __call__(self, population: list[SasSolution], random_state: RandomState) -> list[SasSolution]:
        pass


class Ageing(SolutionSelection):
    """Age the whole population and remove all below a threshold"""

    def __call__(self, population: list[SasSolution], initial_population_size: int, random_state: RandomState) -> list[SasSolution]:
        median_fitness = np.median([i.fitness_ for i in population])
        top_thousand = sorted(population, key=lambda i: i.fitness_, reverse=True)[:1000]
        thousandth_fitness = np.min([i.fitness_ for i in top_thousand])
        for i in range(len(population)):
            population[i].age -= 1
            if population[i].fitness_ < median_fitness: 
                population[i].age += 1
            else:
                population[i].age -= 1
            if len(population) >= 10*initial_population_size and population[i].fitness_ < thousandth_fitness:
                population[i].age -= 2
        return list(i for i in population if i.age > 0)
