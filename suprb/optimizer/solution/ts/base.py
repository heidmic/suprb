from typing import Optional, Union
import copy

import numpy as np

from suprb import Solution
from suprb.solution import SolutionInit, SolutionFitness
from suprb.solution.fitness import NormalizedMOSolutionFitness
from suprb.solution.initialization import RandomInit
from ..base import SolutionComposition, SolutionArchive, MOSolutionComposition, PopulationBasedSolutionComposition
from suprb.utils import check_random_state


class TwoStageSolutionComposition(SolutionComposition):
    """
    Wrapper class for switching between different solution composition algorithms

    Parameters
    ----------
    algorithm_1: PopulationBasedSolutionComposition
        Algorithm to be used in the first stage.
    algorithm_2: PopulationBasedSolutionComposition
        Algorithm to be used in the second stage.
    switch_iteration: int
        Iteration at which to switch from algorithm_1 to algorithm_2.

    """

    def __init__(
        self,
        algorithm_1: PopulationBasedSolutionComposition,
        algorithm_2: PopulationBasedSolutionComposition,
        switch_iteration: int,
        n_iter: int = None,
        init: SolutionInit = RandomInit(),
        archive: SolutionArchive = None,
        random_state: int = None,
        n_jobs: int = None,
        warm_start: bool = True,
        output_fitness: SolutionFitness = NormalizedMOSolutionFitness(),
    ):
        super().__init__(
            n_iter=n_iter, init=init, archive=archive, random_state=random_state, n_jobs=n_jobs, warm_start=warm_start
        )
        self.algorithm_1 = algorithm_1
        self.algorithm_2 = algorithm_2
        self.switch_iteration = switch_iteration

        self.step_ = 0
        self.current_algo_ = algorithm_1

        self.output_fitness = output_fitness

    def optimize(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Union[Solution, list[Solution], None]:
        # This is mega hacky but necessary if the initialisation is to be kept in the suprb fit method
        if self.step_ == 0:
            self.algorithm_1.pool_ = self.pool_
            self.algorithm_2.pool_ = self.pool_
            self.algorithm_1.init.fitness.max_genome_length_ = self.init.fitness.max_genome_length_
            self.algorithm_2.init.fitness.max_genome_length_ = self.init.fitness.max_genome_length_

        self.algorithm_1.random_state = self.random_state
        self.algorithm_2.random_state = self.random_state
        self.random_state_ = check_random_state(self.random_state)

        if self.step_ == self.switch_iteration - 1:
            self.current_algo_ = self.algorithm_2
            if self.warm_start:
                pop = []
                algo_2_fitness = self.algorithm_2.init.fitness
                for solution in self.algorithm_1.population_:
                    solution.fitness = algo_2_fitness
                    solution = self.init.pad(solution, self.random_state_)
                    solution.fit(X, y)
                    pop.append(solution)
                self.algorithm_2.population_ = pop

        self.current_algo_.random_state = self.random_state
        self.current_algo_.optimize(X, y)

        self.population_ = self.current_algo_.population_
        self.step_ += 1

    def elitist(self) -> Optional[Solution]:
        return self.current_algo_.elitist()

    def pareto_front(self) -> list[Solution]:
        if isinstance(self.current_algo_, MOSolutionComposition):
            return self.current_algo_.pareto_front()
        else:
            return [self.current_algo_.elitist()]
