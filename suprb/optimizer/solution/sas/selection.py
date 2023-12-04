from abc import ABCMeta

import numpy as np

from suprb.base import BaseComponent
from .solution_extension import SasSolution
from suprb.utils import RandomState


class SolutionSelection(BaseComponent, metaclass=ABCMeta):

    def __call__(self, population: list[SasSolution], random_state: RandomState) -> list[SasSolution]:
        pass


class Ageing(SolutionSelection):
    """Age the whole population and remove all below a threshold
    Parameters
    ----------
    initial_population_size: int
    top_cutoff_mult: float
        Size multiplier for initial_population_size. If population grows bigger than this age all individuals with worse fitness than the lowest
        in the top initial_population_size * top_cutoff_mult more.
    """

    def __call__(self, population: list[SasSolution], initial_population_size: int, random_state: RandomState, top_cutoff_mult: float = 10) -> list[SasSolution]:
        median_fitness = np.median([i.fitness_ for i in population])
        top_n = initial_population_size * top_cutoff_mult
        top_n_population = sorted(population, key=lambda i: i.fitness_, reverse=True)[:top_n]
        top_n_fitness = top_n_population[-1].fitness_
        for i in range(len(population)):
            population[i].age -= 1
            if population[i].fitness_ > median_fitness: 
                population[i].age += 1
            else:
                population[i].age -= 1
            if len(population) >= top_n and population[i].fitness_ < top_n_fitness:
                population[i].age -= 2
        return list(i for i in population if i.age > 0)
