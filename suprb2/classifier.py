import numpy as np
from suprb2.random_gen import Random
from suprb2.config import Config
from copy import deepcopy


from sklearn.linear_model import LinearRegression
from sklearn.metrics import *

model_list = []


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

    def is_model_in_model_list(self, match):
        global model_list

        for model in model_list:
            if np.array_equal(match, model["match"]):
                return model

        return None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        fits this classifier to the given training date if matched samples changed since the last fit
        :param X:
        :param y:
        :return:
        """
        global model_list

        m = self.matches(X)
        previous_model = self.is_model_in_model_list(m)

        if previous_model:
            self.error = previous_model["error"]
            self.experience = previous_model["experience"]
            self.model = deepcopy(previous_model["model"])
        else:
            self._fit(X[np.nonzero(m)], y[np.nonzero(m)])
            model_list.append({"match": m, "error": self.error, "experience": self.experience, "model": self.model})

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
                self.constant = Classifier.get_default_prediction()
            # TODO is this a good default error? should we use the std?
            #  Equivalent with standardised data?
            self.error = Config().default_error
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
    def random_cl(xdim, *, point=None):
        """
        Returns a randomly placed classifier within [-1, 1]
        If point is given, the classifier bounds will be point +- N(r, r/2)
        with r being defined by Config().rule_discovery['cl_expected_radius']
        Classifiers width is always > 0 in all dimensions
        :param point: center of the classifier
        :return: a new Classifier instance
        """
        if point is None:
            point = Random().random.random(xdim) * 2 - 1
        exp_radius = Config().rule_discovery['cl_expected_radius']
        while True:
            radius = Random().random.normal(loc=exp_radius, scale=exp_radius/2,
                                            size=xdim)
            # emulate do-while loop
            if (radius > 0).all():
                # the probability of a value being below zero is 2.1% as 2.1%
                # of samples fall left of two standard deviations of a normal
                # distribution
                break
        l = np.clip(point - radius, a_min=-1, a_max=1)
        u = np.clip(point + radius, a_min=-1, a_max=1)
        return Classifier(l, u, LinearRegression(), 1)

    def params(self):
        if self.model is LinearRegression:
            return self.model.coef_

    def get_weighted_error(self):
        """
        Calculates the weighted error of the classifier, depending on its error, volume and a constant. 
        -inf is the best possible value for the weighted error
        """
        weighted_error = np.inf
        volume_share = self.get_volume_share()

        if volume_share > 0:
            weighted_error = np.log(self.error) - np.log(volume_share) * \
                Config().rule_discovery["weighted_error_constant"]

        return weighted_error

    def get_volume_share(self):
        """
        Calculates the volume of the classifier in relation to the maximum
        volume of the input space
        :return:
        """
        volume = np.prod(self.upperBounds - self.lowerBounds)
        volume_share = volume / 2 ** len(self.lowerBounds)
        return volume_share

    @staticmethod
    def get_default_prediction():
        return 0.0
