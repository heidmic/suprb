import numpy as np
from joblib import Parallel, delayed

from suprb2.individual import ErrorExperienceHeuristic
from suprb2.optimizer.individual.base import PopulationBasedIndividualOptimizer
from suprb2.optimizer.individual.archive import IndividualArchive
from suprb2.optimizer.individual.fitness import IndividualFitness, ComplexityWu
from suprb2.optimizer.individual.ga.crossover import IndividualCrossover, NPoint
from suprb2.optimizer.individual.ga.mutation import IndividualMutation, BitFlips
from suprb2.optimizer.individual.ga.selection import IndividualSelection, Ranking
from suprb2.optimizer.individual.initialization import IndividualInit, RandomInit


class GeneticAlgorithm(PopulationBasedIndividualOptimizer):
    """ A simple Genetic Algorithm.

    Parameters
    ----------
    n_iter: int
        Iterations the the metaheuristic will perform.
    population_size: int
        Number of individuals in the population.
    mutation: IndividualMutation
    crossover: IndividualCrossover
    selection: IndividualSelection
    fitness: IndividualFitness
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

    def __init__(self,
                 n_iter: int = 128,
                 population_size: int = 128,
                 mutation: IndividualMutation = BitFlips(),
                 crossover: IndividualCrossover = NPoint(),
                 selection: IndividualSelection = Ranking(),
                 fitness: IndividualFitness = ComplexityWu(),
                 init: IndividualInit = RandomInit(mixture=ErrorExperienceHeuristic()),
                 archive: IndividualArchive = None,
                 random_state: int = None,
                 n_jobs: int = 1,
                 warm_start: bool = True,
                 ):
        super().__init__(
            n_iter=n_iter,
            population_size=population_size,
            fitness=fitness,
            init=init,
            archive=archive,
            random_state=random_state,
            n_jobs=n_jobs,
            warm_start=warm_start,
        )

        self.mutation = mutation
        self.crossover = crossover
        self.selection = selection

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        self.fit_population(X, y)

        with Parallel(n_jobs=self.n_jobs) as parallel:
            for _ in range(self.n_iter):
                # Selection
                self.population_ = self.selection(self.population_, self.random_state_)

                # Crossover
                selected = [list(self.random_state_.choice(self.population_, size=2, replace=False)) for _ in
                            range(self.population_size - len(self.population_))]
                crossed = parallel(delayed(self.crossover)(*parents, self.random_state_) for parents in selected)
                self.population_.extend(crossed)

                # Mutation
                self.population_ = parallel(
                    delayed(self.mutation)(individual, self.random_state_) for individual in self.population_)
                self.fit_population(X, y)
