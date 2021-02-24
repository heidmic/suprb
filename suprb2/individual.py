import numpy as np
from suprb2.random_gen import Random
from suprb2.config import Config
from suprb2.classifier import Classifier
from suprb2.pool import ClassifierPool

from sklearn.metrics import *


class Individual:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = None
        self.error = None

    @staticmethod
    def random_individual(genome_length):
        pool_length = len(ClassifierPool().classifiers)
        # from interval [low, high)
        return Individual(np.concatenate((
            Random().random.integers(low=0, high=2, size=pool_length,
                                        dtype=bool),
            np.zeros(genome_length-pool_length))))

    def fit(self, X, y):
        # Classifiers are already fitted, this should thus do nothing. But to
        # keep in line with sklearn syntax is kept in for the moment.
        raise NotImplementedError()

    def predict(self, X):
        out = np.repeat(Config().default_prediction, len(X))
        if X.ndim == 2:
            y_preds = np.zeros(len(X))
            tausum = np.zeros(len(X))
            cls = self.get_classifiers()
            t_ = np.zeros(len(cls))
            for i in range(len(cls)):
                cl = cls[i]
                # unbiased version, with a potential division by zero: 1/(cl.experience - Config().xdim) * cl.error
                with np.errstate(divide='ignore'):
                    tau = 1 / (1 / np.array(cl.experience) * np.array(cl.error))
                t_[i] = tau
            was_inf = np.inf in t_
            for i in range(len(cls)):
                cl = cls[i]
                # an empty array to put predictions in
                local_pred = np.zeros(len(X))
                if was_inf and t_[i] == np.inf:
                    # TODO it might be useful to also do some experience
                    #  weighting here.
                    # TODO Currently a 0 error solution
                    #  fully dominates an e-100 error solution
                    t_[i] = 1
                elif was_inf:
                    t_[i] = 0
                m = cl.matches(X)
                if not m.any():
                    continue
                # put predictions for matched samples into local_pred
                np.put(local_pred, np.nonzero(m), cl.predict(X[np.nonzero(m)])
                       * t_[i])
                # add to the aggregated predictions
                y_preds += local_pred

                local_taus = np.zeros(len(X))
                np.put(local_taus, np.nonzero(m), t_[i])
                tausum += local_taus

            # prevent division by zero
            np.put(tausum, (tausum == 0).nonzero(), 1)

            y_pred = y_preds / tausum
            np.put(out, np.nonzero(y_pred), y_pred)
        # TODO is this shape still needed?
        return out.reshape((-1, 1))

    def get_classifiers(self):
        # TODO for some reason returns tuple here, although this should
        #  only happen for matrizes, see below
        return [ClassifierPool().classifiers[i] for i in np.nonzero(
            self.genome)[0]]

    def determine_fitness(self, X_val, y_val):
        if Config().solution_creation['fitness'] == "pseudo-BIC":
            n = len(X_val)
            # mse = ResidualSumOfSquares / NumberOfSamples
            mse = np.sum(np.square(y_val - self.predict(X_val))) / n
            # for debugging
            self.error = mse
            # BIC -(n * np.log(rss / n) + complexity * np.log(n))
            self.fitness = - (n * np.log(mse) + self.parameters() * np.log(n))

        elif Config().solution_creation['fitness'] == "BIC_matching_punishment":
            n = len(X_val)
            # mse = ResidualSumOfSquares / NumberOfSamples
            mse = np.sum(np.square(y_val - self.predict(X_val))) / n
            # for debugging
            self.error = mse
            matching_pun = np.sum(np.nonzero(np.sum(np.array([cl.matches(X_val)
                                                              for cl
                                                              in
                                                              self.classifiers]),
                                                    1) > 1))
            # BIC -(n * np.log(rss / n) + complexity * np.log(n))
            self.fitness = - (n * np.log(mse) + (self.parameters()
                                                 + matching_pun
                                                 ) * np.log(n))

        elif Config().solution_creation['fitness'] == "MSE":
            self.error = mean_squared_error(y_val, self.predict(X_val))
            self.fitness = - self.error

        elif Config().solution_creation['fitness'] == "simplified_compl":
            self.error = mean_squared_error(y_val, self.predict(X_val))
            self.fitness = - self.error - (len(self.classifiers) - Config().ind_size if len(self.classifiers) > Config().ind_size else 0)

    def parameters(self, simple=True) -> float:
        if simple:
            return np.count_nonzero(self.genome)
        else:
            raise NotImplementedError()
            ## pycharm gives a warning of type missmatch however this seems
            ## to work
            #return np.sum([cl.params() for cl in self.classifiers])

    def mutate(self, rate=0.2):
        pool_length = len(ClassifierPool().classifiers)
        mutations = np.concatenate((Random().random.random(pool_length) < rate,
                        np.zeros(len(self.genome)-pool_length, dtype='bool')),
                                   dtype='bool')
        self.genome = np.logical_xor(self.genome, mutations)
