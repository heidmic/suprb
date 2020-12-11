import numpy as np
from suprb2.random_gen import Random
from suprb2.config import Config
from suprb2.classifier import Classifier

from sklearn.metrics import *


class Individual:
    def __init__(self, classifiers):
        self.classifiers = classifiers
        self.fitness = None
        self.error = None

    @staticmethod
    def random_individual(size):
        return Individual(list(map(lambda x: Classifier.random_cl(),
                                   range(size))))

    def fit(self, X, y):
        # TODO Add note that this does only fit the local models and not optimize classifier location
        for cl in self.classifiers:
            cl.fit(X, y)

    def predict(self, X):
        out = np.repeat(Config().default_prediction, len(X))
        if X.ndim == 2:
            y_preds = np.zeros(len(X))
            taus = np.zeros(len(X))
            for cl in self.classifiers:
                m = cl.matches(X)
                if not m.any():
                    continue
                # an empty array to put predictions in
                local_pred = np.zeros(len(X))
                # unbiased version, with a potential division by zero: 1/(cl.experience - Config().xdim) * cl.error
                tau = 1 / (#1 / (cl.experience + np.finfo(np.float64).tiny) *
                           cl.error + np.finfo(np.float64).tiny)
                # put predictions for matched samples into local_pred
                np.put(local_pred, np.nonzero(m), cl.predict(X[np.nonzero(m)]) * tau)
                # add to the aggregated predictions
                y_preds += local_pred

                local_taus = np.zeros(len(X))
                np.put(local_taus, np.nonzero(m), tau)
                taus += local_taus

            # prevent division by zero
            np.put(taus, (taus == 0).nonzero(), 1)

            y_pred = y_preds / taus
            np.put(out, np.nonzero(y_pred), y_pred)
        # TODO is this shape still needed?
        return out.reshape((-1, 1))

    def determine_fitness(self, X_val, y_val):
        if Config().fitness == "pseudo-BIC":
            n = len(X_val)
            # mse = ResidualSumOfSquares / NumberOfSamples
            mse = np.sum(np.square(y_val - self.predict(X_val))) / n
            # for debugging
            self.error = mse
            # BIC -(n * np.log(rss / n) + complexity * np.log(n))
            self.fitness = - (n * np.log(mse) + self.parameters() * np.log(n))

        elif Config().fitness == "BIC_matching_punishment":
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

        elif Config().fitness == "MSE":
            self.error = mean_squared_error(y_val, self.predict(X_val))
            self.fitness = - self.error

        elif Config().fitness == "stupid_compl":
            self.error = mean_squared_error(y_val, self.predict(X_val))
            self.fitness = - self.error - (len(self.classifiers) - Config().ind_size if len(self.classifiers) > Config().ind_size else 0)

    def parameters(self, simple=True) -> float:
        if simple:
            return len(self.classifiers)
        else:
            # pycharm gives a warning of type missmatch however this seems to work
            return np.sum([cl.params() for cl in self.classifiers])

    def mutate(self):
        # Remove a random classifier
        # TODO add hyperparameter
        if Random().random.random() > 0.5 and len(self.classifiers) > 1:
            self.classifiers.pop(Random().random.integers(low=0, high=len(self.classifiers)-1))

        # Add classifier
        # TODO add hyperparameter
        if Random().random.random() > 0.5:
            self.classifiers.append(Classifier.random_cl())

        # Mutate classifiers
        for cl in self.classifiers:
            cl.mutate()
