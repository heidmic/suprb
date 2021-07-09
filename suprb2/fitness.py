import numpy as np
from sklearn.metrics import *

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
    elif config_fitness_function == "inverted_macro_f1_score":
        fitness_function = inverted_macro_f1_score
    elif config_fitness_function == "inverted_macro_f1_score_times_C":
        fitness_function = inverted_macro_f1_score_times_C
    else:
        print("Invalid fitness function specified! Exiting..")
        fitness_function = None
        exit()
    return fitness_function


def calculate_bic_error(n, y_val, predicted_X_val, config):
    if config.classifier['local_model'] ==  'logistic_regression':
        # Inverted Macro F1 Score
        return 1 - f1_score(y_true=y_val, y_pred=np.rint(predicted_X_val), average='macro')
    elif config.classifier['local_model'] == 'linear_regression':
        # mse = ResidualSumOfSquares / NumberOfSamples
        return np.sum(np.square(y_val - predicted_X_val)) / n
    else:
        raise NotImplementedError


def calculate_bic_fitness(n, parameters, error, config):
    # BIC -(n * np.log(rss / n) + complexity * np.log(n))
    return -1 * (n * np.log(error) + parameters * np.log(n))


def pseudo_bic(X_val, y_val, individual, config):
    n = len(X_val)
    individual.error = calculate_bic_error(n, y_val, individual.predict(X_val))
    individual.fitness = calculate_bic_fitness(n, individual.parameters(), individual.error)


def bic_matching_punishment(X_val, y_val, individual, config):
    n = len(X_val)
    matching_pun = np.sum(np.nonzero(np.sum(np.array([cl.matches(X_val)
                                                      for cl in individual.get_classifiers()]), 1) > 1))

    individual.error = calculate_bic_error(n, y_val, individual.predict(X_val))
    individual.fitness = calculate_bic_fitness(n, individual.parameters() + matching_pun, individual.error)


def mse(X_val, y_val, individual, config):
    individual.error = mean_squared_error(y_val, individual.predict(X_val))
    individual.fitness = -1 * individual.error


def mse_times_C(X_val, y_val, individual, config):
    individual.error = mean_squared_error(y_val, individual.predict(X_val))
    error = individual.error
    if error < config.solution_creation["fitness_target"]:
        error = config.solution_creation["fitness_target"]
    individual.fitness = -1 * error * individual.parameters()


def mse_times_root_C(X_val, y_val, individual, config):
    individual.error = mean_squared_error(y_val, individual.predict(X_val))
    error = individual.error
    if error < config.solution_creation["fitness_target"]:
        error = config.solution_creation["fitness_target"]
    individual.fitness = -1 * error * np.power(individual.parameters(),
        1 / config.solution_creation["fitness_factor"])


def inverted_macro_f1_score(X_val, y_val, individual, config):
    y_pred = individual.predict(X_val)
    individual.fitness = f1_score(y_val, np.rint(y_pred), average='macro')
    individual.error = 1 - individual.fitness


def inverted_macro_f1_score_times_C(X_val, y_val, individual, config):
    y_pred = individual.predict(X_val)
    individual.fitness = f1_score(y_val, np.rint(y_pred), average='macro')
    individual.error = (1 - individual.fitness) * individual.parameters()
