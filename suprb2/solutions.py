from abc import ABC, abstractmethod
from suprb2.random_gen import Random
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
    def __init__(self, X_val, y_val, classifier_pool, config, individual=None):
        self.mutation_rate = config['mutation_rate']
        self.steps = config['steps_per_step']
        self.classifier_pool = classifier_pool
        self.fitness_function_name = config["fitness_function_name"]

        if individual is not None:
            self.individual = individual
        else:
            # This makes most sense when the optimizer is initialised at the
            # start. If a later init is desired, adjust accordingly
            self.individual = Individual.random_individual(
                config["initial_genome_length"], self.classifier_pool, self.fitness_function_name)
            self.individual.determine_fitness(X_val, y_val)

    def step(self, X_val, y_val):
        """
        Create a new solution (global model) by performing multiple
        optimization steps
        :return:
        """
        success = 0
        for i in range(self.steps):
            candidate = Individual(np.copy(self.individual.genome), self.classifier_pool,
                                   self.fitness_function_name)
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
