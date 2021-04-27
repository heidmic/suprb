import math
import numpy as np
from suprb2.random_gen import Random
from suprb2.config import Config

from sklearn.linear_model import LinearRegression
from sklearn.metrics import *


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
        m = ((l <= X) & (X <= u)).all(1)
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
            #  Equivalent with standardised data?
            self.error = Config().var
        else:
            self.model.fit(X, y)
            # TODO should this be on validation data?
            #  We need the score to estimate performance of the whole individual. using validation data would cause an additional loop
            #  using validation data might cause it to bleed over, we should avoid it. in XCS train is used here
            self.error = self.score(X, y, metric=mean_squared_error)
            if self.error <= 1e-4:
                self.error = 1e-4
        self.experience = len(y)

    def score(self, X: np.ndarray, y: np.ndarray, metric=None) -> float:
        if metric is None:
            return self.model.score(X, y)
        else:
            return metric(y, self.predict(X))
            # return metric(y, list(map(self.predict, X)))

    # TODO increasing the probability to expand rather than shrink,
    #  might improve the overall finding of good rules
    def mutate(self, sigma=0.2):
        """
        Mutates this matching function.

        This is done similar to how the first XCSF iteration used mutation
        (Wilson, 2002) but using a Gaussian distribution instead of a uniform
        one (as done by Drugowitsch, 2007): Each interval [l, u)'s bound x is
        changed to x' ~ N(x, (u - l) / 10) (Gaussian with standard deviation a
        10th of the interval's width).
        """
        lowers = Random().random.normal(loc=self.lowerBounds, scale=sigma, size=len(self.lowerBounds))
        uppers = Random().random.normal(loc=self.upperBounds, scale=sigma, size=len(self.upperBounds))
        lu = np.clip(np.sort(np.stack((lowers, uppers)), axis=0), a_max=1, a_min=-1)
        self.lowerBounds = lu[0]
        self.upperBounds = lu[1]

    @staticmethod
    def random_cl(point=None):
        if point is not None:
            lu = np.sort(Random().random.normal(loc=point, scale=2/10, size=(2, Config().xdim)) * 2 - 1, axis=0)
        else:
            lu = np.sort(Random().random.random((2, Config().xdim)) * 2 - 1, axis=0)
        if Config().rule_discovery['cl_min_range']:
            diff = lu[1] - lu[0]
            lu[0] -= diff/2
            lu[1] += diff/2
            lu = np.clip(lu, a_max=1, a_min=-1)
        return Classifier(lu[0], lu[1], LinearRegression(), 1)

    def params(self):
        if self.model is LinearRegression:
            return self.model.coef_

    def get_weighted_error(self):
        '''
        Calculates the weighted error of the classifier, depending on its error, volume and a constant. 
        -inf is the best possible value for the weighted error
        '''
        weighted_error = math.inf
        volume = np.prod(self.upperBounds - self.lowerBounds)

        if volume != 0:
            weighted_error = self.error / (volume * Config().rule_discovery["weighted_error_constant"])

        return weighted_error
