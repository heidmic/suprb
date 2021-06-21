import numpy as np
from sklearn.metrics import *
from suprb2.config import Config

# Solution creation maximizes Fitness of Individuals

def set_fitness_function(config_fitness_function):
    if config_fitness_function == "pseudo-BIC":
        fitness_function = pseudo_bic
    elif config_fitness_function == "BIC_matching_punishment":
        fitness_function = bic_matching_punishment
    elif config_fitness_function == "mse":
        fitness_function = mse
    elif config_fitness_function == "mse_times_C":
        fitness_function = mse_times_C
    elif config_fitness_function == "mse_times_root_C":
        fitness_function = mse_times_root_C
    elif config_fitness_function == "mse_matching_pun":
        fitness_function = mse_matching_pun
    elif config_fitness_function == "mse_times_root_C_matching_pun":
        fitness_function = mse_times_root_C_matching_pun
    else:
        print("Invalid fitness function specified! Exiting..")
        fitness_function = None
        exit()
    return fitness_function


def _calculate_bic_error(n, y_val, predicted_X_val):
    # mse = ResidualSumOfSquares / NumberOfSamples
    return np.sum(np.square(y_val - predicted_X_val)) / n


def _calculate_bic_fitness(n, parameters, error):
    # BIC -(n * np.log(rss / n) + complexity * np.log(n))
    return -1 * (n * np.log(error) + parameters * np.log(n))


def pseudo_bic(X_val, y_val, individual):
    n = len(X_val)
    individual.error = _calculate_bic_error(n, y_val, individual.predict(X_val))
    individual.fitness = _calculate_bic_fitness(n, individual.parameters(), individual.error)


def bic_matching_punishment(X_val, y_val, individual):
    n = len(X_val)
    matching_pun = np.sum(np.nonzero(np.sum(np.array([cl.matches(X_val)
                                                      for cl in individual.get_classifiers()]), 1) > 1))

    individual.error = _calculate_bic_error(n, y_val, individual.predict(X_val))
    individual.fitness = _calculate_bic_fitness(n, individual.parameters() + matching_pun, individual.error)


def mse(X_val, y_val, individual):
    individual.error = mean_squared_error(y_val, individual.predict(X_val))
    individual.fitness = -1 * individual.error


def _cap_error(error):
    if error < Config().solution_creation["fitness_target"]:
        return Config().solution_creation["fitness_target"]
    else:
        return error


def mse_times_C(X_val, y_val, individual):
    individual.error = mean_squared_error(y_val, individual.predict(X_val))
    individual.fitness = -1 * _cap_error(individual.error) * individual.parameters()


def mse_times_root_C(X_val, y_val, individual):
    individual.error = mean_squared_error(y_val, individual.predict(X_val))
    individual.fitness = -1 * _cap_error(individual.error) * np.power(individual.parameters(),
                                                                      1 / Config().solution_creation["fitness_factor"])


def _calc_matching_pun(X_val, individual):
    matching_matrix = np.array([cl.matches(X_val) for cl in
                                individual.get_classifiers()])
    cl_number_per_sample = np.sum(matching_matrix, 1)
    # fraction of samples that are matched by more than
    # fitness_factor classifiers
    return np.sum(cl_number_per_sample >
                  Config().solution_creation[
                      'fitness_factor']) / len(X_val)


def mse_matching_pun(X_val, y_val, individual):
    matching_pun = _calc_matching_pun(X_val, individual)

    individual.error = mean_squared_error(y_val, individual.predict(X_val))
    individual.fitness = -1 * _cap_error(individual.error) * (1 + matching_pun)


def mse_times_root_C_matching_pun(X_val, y_val, individual):
    matching_pun = _calc_matching_pun(X_val, individual)

    individual.error = mean_squared_error(y_val, individual.predict(X_val))
    individual.fitness = -1 * _cap_error(individual.error) * (1 + matching_pun) * \
                         np.power(individual.parameters(), 1 /
                                  Config().solution_creation["fitness_factor"])



