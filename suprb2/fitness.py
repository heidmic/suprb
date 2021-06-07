import numpy as np
from sklearn.metrics import *


def set_fitness_function(config_fitness_function):
    if config_fitness_function == "pseudo-BIC":
        fitness_function = pseudo_bic
    elif config_fitness_function == "BIC_matching_punishment":
        fitness_function = bic_matching_punishment
    elif config_fitness_function == "mse":
        fitness_function = mse
    elif config_fitness_function == "simplified_compl":
        fitness_function = simplified_compl
    else:
        print("Invalid fitness function specified! Exiting..")
        fitness_function = None
        exit()
    return fitness_function


def calculate_bic_error(n, y_val, predicted_X_val):
    # mse = ResidualSumOfSquares / NumberOfSamples
    return np.sum(np.square(y_val - predicted_X_val)) / n


def calculate_bic_fitness(n, parameters, error):
    # BIC -(n * np.log(rss / n) + complexity * np.log(n))
    return -1 * (n * np.log(error) + parameters * np.log(n))


def pseudo_bic(X_val, y_val, individual):
    n = len(X_val)
    individual.error = calculate_bic_error(n, y_val, individual.predict(X_val))
    individual.fitness = calculate_bic_fitness(n, individual.parameters(), individual.error)


def bic_matching_punishment(X_val, y_val, individual):
    n = len(X_val)
    matching_pun = np.sum(np.nonzero(np.sum(np.array([cl.matches(X_val)
                                                      for cl in individual.get_classifiers()]), 1) > 1))

    individual.error = calculate_bic_error(n, y_val, individual.predict(X_val))
    individual.fitness = calculate_bic_fitness(n, individual.parameters() + matching_pun, individual.error)


def mse(X_val, y_val, individual):
    individual.error = mean_squared_error(y_val, individual.predict(X_val))
    individual.fitness = -1 * individual.error


def simplified_compl(X_val, y_val, individual):
    mse(X_val, y_val, individual)
    n = len(individual.get_classifiers())

    # TODO: What is Config().ind_size?
    constant = n - Config().ind_size if n > Config().ind_size else 0
    individual.fitness -= constant
