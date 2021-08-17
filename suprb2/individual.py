import numpy as np
from suprb2.random_gen import Random
from suprb2.classifier import Classifier

from sklearn.metrics import *


class Individual:
    def __init__(self, genome, classifier_pool):
        self.genome = genome
        self.fitness = None
        self.error = None
        self.classifier_pool = classifier_pool

    @staticmethod
    def random_individual(genome_length, classifier_pool):
        pool_length = len(classifier_pool)
        # from interval [low, high)
        random_integers = Random().random.integers(low=0, high=2, size=pool_length, dtype=bool)
        random_genome = np.concatenate((random_integers, np.zeros(genome_length-pool_length)))

        return Individual(random_genome, classifier_pool)

    def fit(self, X, y):
        # Classifiers are already fitted, this should thus do nothing. But to
        # keep in line with sklearn syntax is kept in for the moment.
        raise NotImplementedError()

    def calculate_mixing_weights(self, classifiers):
        tau = np.zeros(len(classifiers))

        for i in range(len(classifiers)):
            experience = np.array(classifiers[i].experience)
            error = np.array(classifiers[i].error)

            if experience != 0:
                with np.errstate(divide='ignore'):
                    tau[i] = 1 / ((1 / experience) * error)

        return tau

    def predict(self, X):
        out = np.repeat(Classifier.get_default_prediction(), len(X))
        if X.ndim == 2:
            classifiers = self.get_classifiers()
            y_preds = np.zeros(len(X))
            tausum = np.zeros(len(X))
            t_ = self.calculate_mixing_weights(classifiers)
            was_inf = np.inf in t_
            for i in range(len(classifiers)):
                cl = classifiers[i]
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
            np.putmask(out, y_pred, y_pred)
        # TODO is this shape still needed?
        return out.reshape((-1, 1))

    def get_classifiers(self):
        # TODO for some reason returns tuple here, although this should
        #  only happen for matrizes, see below
        return [self.classifier_pool[i] for i in np.nonzero(self.genome)[0]]

    def parameters(self, simple=True) -> float:
        if simple:
            return np.count_nonzero(self.genome)
        else:
            raise NotImplementedError()
            # pycharm gives a warning of type missmatch however this seems
            # to work
            # return np.sum([cl.params() for cl in self.classifiers])

    def mutate(self, rate=0.2):
        pool_length = len(self.classifier_pool)
        random_value = Random().random.random(pool_length)
        random_genome = np.zeros((len(self.genome)-pool_length), dtype='bool')
        mutations = np.concatenate((random_value < rate, random_genome), dtype='bool')
        self.genome = np.logical_xor(self.genome, mutations)
