import numpy as np
from suprb2.random_gen import Random
from suprb2.config import Config
from suprb2.perf_recorder import PerfRecorder
from suprb2.classifier import Classifier
from suprb2.individual import Individual

from sklearn.model_selection import train_test_split
from datetime import datetime
import mlflow as mf
from copy import deepcopy


class LCS:
    def __init__(self, xdim,
                 # pop_size=30, ind_size=50, generations=50,
                 # fitness="pseudo-BIC",
                 logging=True):
        Config().xdim = xdim
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

    def fit(self, X, y):
        Config().default_prediction = np.mean(y)
        Config().var = np.var(y)
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=Random().split_seed())

        self._train(X_train, y_train, X_val, y_val, gen=0)

        # TODO allow other termination criteria. Early Stopping?
        for i in range(Config().generations):
            # TODO allow more elitist
            elitist = deepcopy(self.elitist)
            new_pop = list()
            while len(new_pop) < Config().pop_size:
                parents = self.tournament_simple(2)
                child_a, child_b = self.crossover(parents[0], parents[1])
                new_pop.append(child_a)
                new_pop.append(child_b)
            self.population = new_pop
            for ind in self.population:
                ind.mutate()
            self._train(X_train, y_train, X_val, y_val, i+1)
            self.population.append(elitist)
            if i % 5 == 0:
                print(f"Finished generation {i+1} at {datetime.now().time()}")

    def _train(self, X_train, y_train, X_val, y_val, gen):
        for ind in self.population:
            ind.fit(X_train, y_train)
            ind.determine_fitness(X_val, y_val)
            # TODO allow more elitist
            if self.elitist is None or self.elitist.fitness < ind.fitness:
                self.elitist = ind
        if Config().logging:
            mf.log_metric("fitness elite", self.elitist.fitness, gen)
            PerfRecorder().elitist_fitness.append(self.elitist.fitness)
            PerfRecorder().elitist_val_error.append(self.elitist.error)
            PerfRecorder().val_size.append(len(X_val))
            PerfRecorder().elitist_matched.append(np.sum(np.array([cl.matches(X_val) for cl in self.elitist.classifiers]).any(axis=0)))
            PerfRecorder().elitist_complexity.append(self.elitist.parameters())

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
            cl = Classifier.random_cl(x)
            cl.fit(X, y)
            for i in range(Config().rule_discovery['steps_per_step']):
                children = list()
                for j in range(Config().rule_discovery['lmbd']):
                    child = deepcopy(cl)
                    child.mutate(Config().rule_discovery['sigma'])
                    child.fit(X, y)
                    children.append(child)
                # code inspection predicts a type missmatch but it should be fine?
                cl = children[np.argmin([child.error for child in children])]
            if cl.error < self.default_error(y[np.nonzero(cl.matches(X))]):
                ClassifierPool().classifiers.append(cl)

    @staticmethod
    def default_error(y):
        return np.sum(y**2)/len(y)


    # place classifiers around those examples
    # test if classifiers overlap
    # remove random overlapping classifiers until overlaps are resolved
    # draw new examples and place classifiers around them until no more
    #   overlaps are found
    ## maybe we allow some overlap? Would make the checks optional (and easier)

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
