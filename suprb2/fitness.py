import numpy as np
from sklearn.metrics import *


class Fitness:
    def __init__(self, config_fitness_function):
        self.fitness_function = None
        self.error = None
        self.fitness = None

        self.set_fitness_function(config_fitness_function)

    def set_fitness_function(self, config_fitness_function):
        if config_fitness_function == "pseudo-BIC":
            self.fitness_function = self.pseudo_bic
        elif config_fitness_function == "BIC_matching_punishment":
            self.fitness_function = self.bic_matching_punishment
        elif config_fitness_function == "mse":
            self.fitness_function = self.mse
        elif config_fitness_function == "simplified_compl":
            self.fitness_function = self.simplified_compl
        else:
            print("Invalid fitness function specified! Exiting..")
            self.fitness_function = None
            exit()

    def calculate_bic_error(self, n, y_val, predicted_X_val):
        # mse = ResidualSumOfSquares / NumberOfSamples
        self.error = np.sum(np.square(y_val - predicted_X_val)) / n

    def calculate_bic_fitness(self, n, parameters):
        # BIC -(n * np.log(rss / n) + complexity * np.log(n))
        self.fitness = - (n * np.log(self.error) + parameters * np.log(n))

    def pseudo_bic(self, X_val, y_val, predicted_X_val, parameters, classifiers):
        n = len(X_val)
        self.calculate_bic_error(n, y_val, predicted_X_val)
        self.calculate_bic_fitness(n, parameters)

    def bic_matching_punishment(self, X_val, y_val, predicted_X_val, parameters, classifiers):
        n = len(X_val)
        matching_pun = np.sum(np.nonzero(np.sum(np.array([cl.matches(X_val) for cl in classifiers]), 1) > 1))
        
        self.calculate_bic_error(n, y_val, predicted_X_val)
        self.calculate_bic_fitness(n, parameters + matching_pun)

    def mse(self, X_val, y_val, predicted_X_val, parameters, classifiers):
        self.error = mean_squared_error(y_val, predicted_X_val)
        self.fitness = -1 * self.error

    # TODO: What is Config().ind_size?
    def simplified_compl(self, X_val, y_val, predicted_X_val, parameters, classifiers):
        self.mse(X_val, y_val, predicted_X_val, parameters, classifiers)

        constant = len(classifiers) - Config().ind_size if len(classifiers) > Config().ind_size else 0
        self.fitness -= constant

    def determine_fitness(self, X_val, y_val, predicted_X_val, parameters, classifiers):
        self.fitness_function(X_val, y_val, predicted_X_val, parameters, classifiers)
