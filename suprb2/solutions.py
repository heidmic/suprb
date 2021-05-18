from abc import ABC, abstractmethod
from suprb2.random_gen import Random
from suprb2.config import Config
from suprb2.individual import Individual
import numpy as np  # type: ignore


class SolutionOptimizer(ABC):

    @abstractmethod
    def step(self, X_val, y_val):
        """
        Create a new solution (global model)
        :return:
        """
        pass

    @abstractmethod
    def get_elitist(self):
        """

        :return: the current solution
        """
        pass


class ES_1plus1(SolutionOptimizer):
    _config = {"name": '(1+1)-ES',
               "mutation_rate": 0.2,
               "fitness": "MSE_matching_pun",
               "fitness_factor": 5,
               "steps_per_step": 100}

    def __init__(self, X_val, y_val, classifier_pool, individual=None):
        self.mutation_rate = Config().solution_creation['mutation_rate']
        self.steps = Config().solution_creation['steps_per_step']
        self.classifier_pool = classifier_pool
        
        if individual is not None:
            self.individual = individual
        else:
            # This makes most sense when the optimizer is initialised at the
            # start. If a later init is desired, adjust accordingly
            self.individual = Individual.random_individual(
                Config().initial_genome_length, self.classifier_pool)
            self.individual.determine_fitness(X_val, y_val)

    def step(self, X_val, y_val):
        """
        Create a new solution (global model) by performing multiple
        optimization steps
        :return:
        """
        success = 0
        for i in range(self.steps):
            candidate = Individual(np.copy(self.individual.genome), self.classifier_pool)
            candidate.mutate(self.mutation_rate)
            candidate.determine_fitness(X_val, y_val)
            if self.individual.fitness < candidate.fitness:
                self.individual = candidate
                success += 1
        return success

    def get_elitist(self):
        """

        :return: the current solution
        """
        return self.individual


from deap import base, creator, tools
import random
from sklearn.metrics import mean_squared_error


class NSGA_II(SolutionOptimizer):
    """
    This roughly follows https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py
    """

    _config = {"name": "NSGA-II",
               "steps_per_step": 100,
               # TODO as per https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.selTournamentDCD
               #  pop_size needs to be divible by four to perform selection
               #  as we were before. should we change this?
               "pop_size": 40,
               "recom_prob": 0.5,
               "recom_rate": 0.2,  # TODO what is a good value here?
               "mut_rate": 0.2,
               "initial_solution_size": 10}

    def __init__(self, X_val, y_val, classifier_pool):
        self.X = X_val
        self.y = y_val
        self.classifier_pool = classifier_pool

        # DEAP uses the global ``random.random`` RNG.
        random.seed(Random().split_seed())
        self.toolbox = base.Toolbox()

        # fitness is comprised of mean_squared_error and complexity
        creator.create("FitnessMin", base.Fitness, weights=(-1., -1.))
        creator.create("Genotype", list, fitness=creator.FitnessMin)

        def _random_genome(pool_length):
            isz = NSGA_II._config["initial_solution_size"]
            classifiers_on_init = Random().random.normal(loc=isz, scale=isz/2)
            classifiers_on_init = int(np.clip(classifiers_on_init, 0, isz*2))
            # from interval [low, high)
            ones = Random().random.integers(low=0, high=pool_length,
                                            size=classifiers_on_init)
            genome = np.zeros(pool_length)
            genome[ones] = 1
            return creator.Genotype(genome)

        self.toolbox.register("genotype", _random_genome,
                              pool_length=len(self.classifier_pool))
        self.toolbox.register("population", tools.initRepeat, list,
                              self.toolbox.genotype)
        self.toolbox.register("prep_select", tools.selNSGA2)
        self.toolbox.register("select", tools.selTournamentDCD)
        self.toolbox.register("recombine", tools.cxUniform)
        self.toolbox.register("mutate", tools.mutFlipBit)

        def _evaluate(genotype, classifier_pool):
            phenotype = Individual(genotype, classifier_pool)
            mse = mean_squared_error(y_val, phenotype.predict(X_val))
            # TODO if complexity is zero, we probably would not want to use
            #  that solution
            return mse, phenotype.parameters()

        self.toolbox.register("evaluate", _evaluate)

        self.pop = self.toolbox.population(n=NSGA_II._config["pop_size"])

        fitnesses = [self.toolbox.evaluate(ind, self.classifier_pool) for ind
                     in self.pop]
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        self.pop = self.toolbox.prep_select(self.pop, len(self.pop))
        NSGA_II._config.update(Config().solution_creation)

    def step(self, X_val, y_val):
        """
        Create a new solution (global model)
        :return:
        """
        if len(self.pop[0]) < len(self.classifier_pool):
            size = len(self.classifier_pool) - len(self.pop[0])
            for i in self.pop:
                i.extend(np.zeros(size))
        for i in range(NSGA_II._config["steps_per_step"]):
            # Vary the population
            offspring = self.toolbox.select(self.pop, len(self.pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= NSGA_II._config["recom_prob"]:
                    self.toolbox.recombine(ind1, ind2, NSGA_II._config[
                        "recom_rate"])

                self.toolbox.mutate(ind1, NSGA_II._config["mut_rate"])
                self.toolbox.mutate(ind2, NSGA_II._config["mut_rate"])
                del ind1.fitness.values, ind2.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = [self.toolbox.evaluate(ind, self.classifier_pool) for
                         ind in invalid_ind]
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            self.pop = self.toolbox.prep_select(self.pop + offspring,
                                                NSGA_II._config["pop_size"])
        pass

    def get_elitist(self):
        """

        :return: the current solution
        """
        elitist = Individual(self.pop[0], self.classifier_pool)
        elitist.error = self.pop[0].fitness.values[0]

        # TODO do we need a better value here? probably
        #  logging should in some way be adjusted
        elitist.fitness = 0

        return elitist






