import numpy as np
from suprb2.random_gen import Random
from suprb2.config import Config
from suprb2.perf_recorder import PerfRecorder
from suprb2.classifier import Classifier
from suprb2.individual import Individual
from suprb2.solutions import ES_1plus1

from sklearn.model_selection import train_test_split
from datetime import datetime
from copy import deepcopy
import mlflow as mf
import itertools


class LCS:
    def __init__(self, xdim,
                 # pop_size=30, ind_size=50, generations=50,
                 # fitness="pseudo-BIC",
                 logging=True):
        self.xdim = xdim
        # Config().pop_size = pop_size
        # Config().ind_size = ind_size
        # Config().generations = generations
        # Config().fitness = fitness
        Config().logging = logging
        if Config().logging:
            mf.log_params(Config().__dict__)
            mf.log_param("seed", Random()._seed)
            self.config = Config()
            self.perf_recording = PerfRecorder()
        self.sol_opt = None
        self.rules_discovery_duration_cumulative = 0
        self.solution_creation_duration_cumulative = 0
        self.classifier_pool = list()

    def calculate_delta_time(self, start_time, end_time):
        delta_time = end_time - start_time
        delta = delta_time.seconds + (delta_time.microseconds / 1e6)
        return round(delta, 3)

    def log_discover_rules_duration(self, start_time, discover_rules_time, step):
        discover_rules_duration = self.calculate_delta_time(start_time, discover_rules_time)
        self.rules_discovery_duration_cumulative += discover_rules_duration
        mf.log_metric("rules_discovery_duration", discover_rules_duration, step)
        mf.log_metric("rules_discovery_duration_cumulative", self.rules_discovery_duration_cumulative, step)

    def log_solution_creation_duration(self, start_time, solution_creation_time, step):
        solution_creation_duration = self.calculate_delta_time(start_time, solution_creation_time)
        self.solution_creation_duration_cumulative += solution_creation_duration
        mf.log_metric("solution_creation_duration", solution_creation_duration, step)
        mf.log_metric("solution_creation_duration_cumulative", self.solution_creation_duration_cumulative, step)

    def run_inital_step(self, X, y):
        start_time = datetime.now()
        while len(self.classifier_pool) < Config().initial_pool_size:
            self.discover_rules(X, y)

        discover_rules_time = datetime.now()

        self.sol_opt = ES_1plus1(X, y, self.classifier_pool)
        # self.sol_opt = ES_1plus1(X_val, y_val)
        solution_creation_time = datetime.now()

        if Config().logging:
            self.log(0, X)
            # self.log(0, X_val)
            self.log_discover_rules_duration(start_time, discover_rules_time, 0)
            self.log_solution_creation_duration(discover_rules_time, solution_creation_time, 0)

    def fit(self, X, y):
        # if Config().use_validation:
        #     X_train, X_val, y_train, y_val = train_test_split(X, y,
        #                                                       random_state=Random().split_seed())
        #
        #
        # else:
        #     X_train = X
        #     X_val = X
        #     y_train = y
        #     y_val = y

        self.run_inital_step(X, y)

        # TODO allow other termination criteria. Early Stopping?
        for step in range(Config().steps):
            start_time = datetime.now()

            self.discover_rules(X, y)
            discover_rules_time = datetime.now()

            self.sol_opt.step(X, y)
            # self.sol_opt.step(X_val, y_val)
            solution_creation_time = datetime.now()

            if Config().logging:
                self.log(step+1, X)
                # self.log(0, X_val)
                self.log_discover_rules_duration(start_time, discover_rules_time, step+1)
                self.log_solution_creation_duration(discover_rules_time, solution_creation_time, step+1)

            # add verbosity option
            if step % 25 == 0:
                print(f"Finished step {step + 1} at {datetime.now().time()}\n")

    def log(self, step, X_val):
        mf.log_metric("fitness elite", self.sol_opt.get_elitist()
                      .fitness, step)
        mf.log_metric("error elite", self.sol_opt.get_elitist()
                      .error, step)
        mf.log_metric("complexity elite", self.sol_opt.get_elitist()
                      .parameters(), step)
        mf.log_metric("classifier pool size", len(self.classifier_pool),
                      step)
        PerfRecorder().elitist_fitness.append(
            self.sol_opt.get_elitist().fitness)
        PerfRecorder().elitist_val_error.append(
            self.sol_opt.get_elitist().error)
        PerfRecorder().val_size.append(len(X_val))
        PerfRecorder().elitist_matched.append(np.sum(np.array(
            [cl.matches(X_val) for cl in
             [self.classifier_pool[i] for i in np.nonzero(
                 self.sol_opt.get_elitist().genome)[0]]]).any(axis=0)))
        PerfRecorder().elitist_complexity.append(
            self.sol_opt.get_elitist().parameters())

    def get_elitist(self):
        return self.sol_opt.get_elitist()

    def predict(self, X):
        return self.sol_opt.get_elitist().predict(X)

    def score(self, X, y):
        # TODO add a score according to https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects
        #  likely R2
        raise NotImplementedError()

    def discover_rules(self, X, y):
        # draw n examples from data
        idxs = Random().random.choice(np.arange(len(X)),
                                      Config().rule_discovery['nrules'], False)

        for x in X[idxs]:
            cl = Classifier.random_cl(self.xdim, point=x)
            cl.fit(X, y)
            for i in range(Config().rule_discovery['steps_per_step']):
                children = list()
                for j in range(Config().rule_discovery['lmbd']):
                    child = deepcopy(cl)
                    child.mutate(Config().rule_discovery['sigma'])
                    child.fit(X, y)
                    children.append(child)
                # ToDo instead of greedily taking the minimum, treating all
                #  below a certain threshhold as equal might yield better models
                cl = children[np.argmin([child.get_weighted_error() for child in children])]

            if cl.error < self.default_error(y[np.nonzero(cl.matches(X))]):
                self.classifier_pool.append(cl)

    @staticmethod
    def default_error(y):
        # for standardised data this should be equivalent to np.var(y)
        with np.errstate(invalid='ignore'):
            return np.sum(y**2)/np.array(len(y))

    # place classifiers around those examples
    # test if classifiers overlap
    # remove random overlapping classifiers until overlaps are resolved
    # draw new examples and place classifiers around them until no more
    #   overlaps are found
    # maybe we allow some overlap? Would make the checks optional (and easier)

    # fit all classifiers

    #

    # do
    #   determine subpopulation based on localisation
    #   perform a few generations of ES/GA for each subpop
    #       (maybe limit this based on local solution strength
    #   select good solutions and make them available for LCS
    #       set aside/freeze or do we just hope they survive?
    # while LCS termination criterion is not found


# def crossover(self, parent_a, parent_b):
#     """
#     Creates offspring from the two given individuals by crossover.
#
#     – If `crossover_type` is `"normal"` for this population, we use a
#       normal distribution to determine sizes of the children to more often
#       create similarly large children.
#
#     – If `crossover_type` is `"uniform"`, the crossover definition from
#       (Drugowitsch, 2007) is used (e.g. creates one very small and one very
#       big child with the same probability as it does two children of equal
#       size).
#
#     – If `crossover_type` is `"off"`, no crossover is performed but (deep)
#       copies of the parents are returned.
#
#     If the parent's combined lengths are less than 2, return the parents
#     unchanged.
#     """
#     if len(parent_a.classifiers) + len(parent_b.classifiers) <= 2:
#         return parent_a, parent_b
#     elif Config().crossover_type == "off":
#         return deepcopy(parent_a), deepcopy(parent_b)
#     else:
#         parenta_cl = deepcopy(parent_a.classifiers)
#         parentb_cl = deepcopy(parent_b.classifiers)
#         all_cl = parenta_cl + parentb_cl
#
#         Random().random.shuffle(all_cl)
#
#         if Config().crossover_type == "normal":
#             p = 0
#             # ensure there is at least one classifier per individuum
#             while not 1 <= p <= len(all_cl) - 1:
#                 # random crossover point
#                 p = round(Random().random.normal(loc=np.floor(len(all_cl) / 2)))
#         elif Config().crossover_type == "uniform":
#             p = Random().random.integers(1, len(all_cl) - 1)
#
#         cl_a_new = all_cl[:p]
#         cl_b_new = all_cl[p:]
#
#         return Individual(cl_a_new), Individual(cl_b_new)
#
#
# def tournament_simple(self, n: int, size: int = 2):
#     """
#     Very simple tournament selection: Each tournament only consists of two
#     individuals the best of which always wins.
#
#     Note: We select competitors without replacement.
#
#     :param n: How many individuals to select (by holding successive
#         tournaments).
#     :param size: Tournament size, default is 2
#     """
#     winners = list()
#     for _ in range(n):
#         competitors = Random().random.choice(self.population,
#                                              size=size,
#                                              replace=False)
#         # if competitors[0].fitness > competitors[1].fitness:
#         #   winners.append(competitors[0])
#         # else:
#         #    winners.append(competitors[1])
#         winners.append(competitors[np.argmax([ind.fitness for ind in
#                                               competitors])])
#     return winners


if __name__ == '__main__':
    pass
