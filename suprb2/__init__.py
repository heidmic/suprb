import numpy as np
from copy import deepcopy
from suprb2.random_gen import Random
from suprb2.config import Config
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class Classifier:
    def __init__(self, lowers, uppers, local_model, degree):
        self.lowerBounds = lowers
        self.upperBounds = uppers
        self.model = local_model
        # TODO make this part of local model (as a class)
        self.degree = degree
        self.error = None
        # TODO expand this int into remember which was matched (to reduce
        # retraining when mutate didnt change the matched data)
        self.experience = None
        # if set this overrides local_model and outputs constant for all prediction requests
        self.constant = None
        self.last_training_match = None

    def matches(self, X: np.array) -> np.array:
        l = np.reshape(np.tile(self.lowerBounds, X.shape[0]), (X.shape[0],
                                                               X.shape[1]))
        u = np.reshape(np.tile(self.upperBounds, X.shape[0]), (X.shape[0],
                                                               X.shape[1]))
        # Test if greater lower and smaller upper and return True for each line
        m = ((l < X) & (X < u)).all(1)
        return m

    def predict(self, X: np.ndarray) -> float:
        """
        Returns this classifier's prediction for the given input.
        """
        if X.ndim == 2:
            if self.constant is None:
                return self.model.predict(X)
            else:
                return np.repeat(self.constant, X.shape[0])
        else:
            if self.constant is None:
                # TODO why the last reshape?
                return self.model.predict(X.reshape((1, -1))).reshape(())
            else:
                return self.constant

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        fits this classifier to the given training date if matched samples changed since the last fit
        :param X:
        :param y:
        :return:
        """
        m = self.matches(X)
        # we save the training match to optimize fit (only re-fit when match changed after mutation)
        if self.last_training_match is None or (m != self.last_training_match).any():
            self.last_training_match = m
            self._fit(X[np.nonzero(m)], y[np.nonzero(m)])

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits this classifier to the given training data and computes its
        error on it using it's local model's score method.
        """
        self.constant = None
        if len(X) < 2:
            if len(X) == 1:
                self.constant = y[0]
            else:
                self.constant = Config().default_prediction
            # TODO is this a good default error? should we use the std?
            self.error = Config().var
        else:
            self.model.fit(X, y)
            # TODO should this be on validation data?
            #  We need the score to estimate performance of the whole individual. using validation data would cause an additional loop
            #  using validation data might cause it to bleed over, we should avoid it. in XCS train is used here
            self.error = self.score(X, y, metric=mean_squared_error)
        self.experience = len(y)

    def score(self, X: np.ndarray, y: np.ndarray, metric=None) -> float:
        if metric is None:
            return self.model.score(X, y)
        else:
            return metric(y, self.predict(X))
            #return metric(y, list(map(self.predict, X)))

    def mutate(self):
        """
        Mutates this matching function.

        This is done similar to how the first XCSF iteration used mutation
        (Wilson, 2002) but using a Gaussian distribution instead of a uniform
        one (as done by Drugowitsch, 2007): Each interval [l, u)'s bound x is
        changed to x' ~ N(x, (u - l) / 10) (Gaussian with standard deviation a
        10th of the interval's width).
        """
        lowers = Random().random.normal(loc=self.lowerBounds, scale=2/10, size=len(self.lowerBounds))
        uppers = Random().random.normal(loc=self.upperBounds, scale=2/10, size=len(self.upperBounds))
        lu = np.clip(np.sort(np.stack((lowers, uppers)), axis=0), a_max=1, a_min=-1)
        self.lowerBounds = lu[0]
        self.upperBounds = lu[1]

    @staticmethod
    def random_cl():
        lu = np.sort(Random().random.random((2, Config().xdim)) * 2 - 1, axis=0)
        if Config().cl_min_range:
            diff = lu[1] - lu[0]
            lu[0] -= diff/2
            lu[1] += diff/2
            lu = np.clip(lu, a_max=1, a_min=-1)
        return Classifier(lu[0], lu[1], LinearRegression(), 2)

    def params(self):
        if self.model is LinearRegression:
            return self.model.coef_


class Individual:
    def __init__(self, classifiers):
        self.classifiers = classifiers
        self.fitness = None

    @staticmethod
    def random_individual(size):
        return Individual(list(map(lambda x: Classifier.random_cl(),
                                   range(size))))

    def fit(self, X, y):
        # TODO Add note that this does only fit the local models and not optimize classifier location
        for cl in self.classifiers:
            cl.fit(X, y)

    def predict(self, X):
        # TODO make this better
        out = np.repeat(Config().default_prediction, X.shape[0])
        if X.ndim == 2:
            for x_idx in range(X.shape[0]):
                x = X[x_idx]
                mixing_sum = 0
                mixing_taus = 0
                for cl in self.classifiers:
                    m = cl.matches(x.reshape((1, -1)))
                    if not m.any():
                        continue
                    mixing_sum += cl.predict(X[np.nonzero(m)])[0]
                    # see drugowich 6.26
                    mixing_taus += 1/(cl.experience - Config().xdim) * cl.error
                if not mixing_sum == 0:
                    out[x_idx] = mixing_sum / mixing_taus
        return out.reshape((-1, 1))

    def determine_fitness(self, X_val, y_val):
        n = len(X_val)
        # BIC -(n * np.log(rss / n) + complexity * np.log(n))
        self.fitness = -(n * np.log(np.sum(np.square(y_val - self.predict(X_val))) / n)
                         + self.parameters() * np.log(n))

    def parameters(self, simple=True) -> float:
        if simple:
            return len(self.classifiers)
        else:
            # pycharm gives a warning of type missmatch however this seems to work
            return np.sum([cl.params() for cl in self.classifiers])

    def mutate(self):
        # Add classifier
        # TODO add hyperparameter
        if Random().random.random() > 0.5:
            self.classifiers.append(Classifier.random_cl())

        # Remove a random classifier
        # TODO add hyperparameter
        if Random().random.random() > 0.5:
            self.classifiers.pop(Random().random.integers(len(self.classifiers)))

        # Mutate classifiers
        for cl in self.classifiers:
            cl.mutate()


class LCS:
    def __init__(self, xdim, individuals=None, cl_min_range=None, pop_size=30, ind_size=50, generations=50):
        Config().xdim = xdim
        Config().cl_min_range = cl_min_range
        Config().pop_size = pop_size
        Config().ind_size = ind_size
        Config().generations = generations
        if individuals is None:
            self.population = (list(map(lambda x: Individual.random_individual(
                Config().ind_size), range(Config().pop_size))))
        else:
            self.population = individuals
        self.elitist = None

    def fit(self, X, y):
        Config().default_prediction = np.mean(y)
        Config().var = np.var(y)
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=Random().split_seed())

        self._train(X_train, y_train, X_val, y_val)

        # TODO allow other termination criteria. Early Stopping?
        for i in range(Config().generations):
            # TODO allow more elitist
            # TODO currently population grows by 1 every generation
            elitist = deepcopy(self.elitist)
            for ind in self.population:
                ind.mutate()
            self._train(X_train, y_train, X_val, y_val)
            self.population.append(elitist)

    def _train(self, X_train, y_train, X_val, y_val):
        for ind in self.population:
            ind.fit(X_train, y_train)
            ind.determine_fitness(X_val, y_val)
            # TODO allow more elitist
            if self.elitist is None or self.elitist.fitness < ind.fitness:
                self.elitist = ind

    def predict(self, x):
        raise NotImplementedError()


if __name__ == '__main__':
    pass
