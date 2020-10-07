"""
General problem interface.

Input values to the quality function are assumed to be from [0, 1]^(x*a) and
are uniformly sampled by default. In order to sample from a different region
from X and A (or, e.g., not uniformly), overwrite `gen_x` and `gen_a`,
respectively.
"""

import sys
from abc import *
from datetime import datetime
from typing import Callable, Tuple

import numpy as np  # type: ignore
from scipy.optimize import *
from sklearn.metrics import *


class Problem(ABC):
    """
    :param q: Quality function taking NumPy arrays of shape `(xdim + adim, )`
        to `float`. This is being sampled in the interval `[-1, 1]^(xdim +
        adim)`, so make sure to scale accordingly.
    """
    def __init__(self, seed: int, xdim: int, adim: int):
        self.xdim: int = xdim
        self.adim: int = adim
        self.seed: int = seed
        self.random = np.random.default_rng(seed)
        # TODO We sample a seed from self.random for now, is this OK?
        self._random_sample = np.random.default_rng(
            self.random.integers(999999999))
        # TODO Should xmin and xmax be vectors to restrict randomness
        # dimension-wise?

    @abstractmethod
    def q(xa: np.ndarray) -> float:
        pass

    def gen_xa(self) -> np.ndarray:
        return self._random_sample.uniform(-1, 1, self.xdim + self.adim)

    def generate(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        XA = np.array(list(map(lambda x: self.gen_xa(), range(n))))
        # TODO use this instead!
        # XA = self._random_sample.uniform(-1, 1, … (n, self.xdim))
        q = np.array(list(map(self.q, XA)))
        return XA, q

    def error_choice(self, X_eval: np.ndarray, q_opt_eval: np.ndarray,
                     A_eval_: np.ndarray, *metrics) -> np.ndarray:
        """
        MSE between (real) quality at real optimal action and (real) quality at
        predicted optimal action.

        :param X_eval: Points where choice was made.
        :param q_opt_eval: Quality of actual optimal choices.
        :param A_eval_: Predictions for optimal choices.
        """
        if metrics is None or len(metrics) == 0:
            metrics = [mean_squared_error]

        # prediction
        XA_eval_ = join_XA(X_eval, A_eval_)

        # actual qualities of prediction and ground truth
        q_eval_ = np.array(list(map(self.q, XA_eval_)))

        return np.array(
            list(map(lambda metric: metric(q_opt_eval, q_eval_), metrics)))

    def highest_q_in_interval(self):
        """
        Estimates the highest quality by sampling a fixed number of times
        (10000 for now).
        """
        # TODO Add hyper parameter for “10000” (although it doesn't really matter)
        _, q = self.generate(10000)
        return np.max(q)

    def highest_q_in_interval_diff_evo(self):
        bounds = np.stack((-1 * np.ones(self.adim), np.ones(self.adim))). \
            T.reshape(self.adim, 2)

        def inv_q(xa):
            return -self.q(xa)

        return -differential_evolution(inv_q, bounds=bounds).fun

    def _q_to_optimize_a(self, a: np.ndarray, *args) -> float:
        """
        This function allows to call a minimizer on q that provides the best
        attainable solution for the a dimension
        :param a: point where minimiser probes
        :param args: x vector where to optimise (contained in a tuple)
        :return: q(x,a)
        """
        x = args[0]
        xa = np.append(x, a)
        return -self.q(xa)

    # TODO should we rename this?
    # TODO refactoring
    def optimal_choice(self, X_pois: np.ndarray, optimizer="sample"):
        """
        Maximizes problems function for given x coordinates and returns the
        highest attainable q value when using the optimizer specified.
        Available optimizers are:
            "loc" gradient based local optimum search
            "DE" differential evolution
            "BH" basin hopping
            "DA" dual annealing
        :param X_pois: array of x coordinates for which to maximize
        :param optimizer: strategy to employ (see description for options)
        :return: maximum q value
        """
        if optimizer == "loc":
            return self._optimal_choice_loc(X_pois) * -1
        if optimizer == "DE":
            return self._optimal_choice_de(X_pois) * -1
        if optimizer == "BH":
            return self._optimal_choice_bh(X_pois) * -1
        if optimizer == "DA":
            return self._optimal_choice_da(X_pois) * -1
        if optimizer == "sample":
            return self._optimal_choice_sample(X_pois)

    def _optimal_choice_loc(self, X_pois: np.ndarray):
        bounds = np.stack((-1 * np.ones(self.adim), np.ones(self.adim))). \
            T.reshape(self.adim, 2)
        return np.array(
            list(
                map(
                    lambda poi: minimize(self._q_to_optimize_a,
                                         np.zeros(self.adim),
                                         args=(poi, ),
                                         bounds=bounds).fun, X_pois)))

    def _optimal_choice_de(self, X_pois: np.ndarray):
        bounds = np.stack((-1 * np.ones(self.adim), np.ones(self.adim))). \
            T.reshape(self.adim, 2)
        return np.array(
            list(
                map(
                    lambda poi: differential_evolution(self._q_to_optimize_a,
                                                       args=(poi, ),
                                                       bounds=bounds).fun,
                    X_pois)))

    def _optimal_choice_bh(self, X_pois: np.ndarray):
        bounds = np.stack((-1 * np.ones(self.adim), np.ones(self.adim))). \
            T.reshape(self.adim, 2)
        return np.array(
            list(
                map(
                    lambda poi: basinhopping(self._q_to_optimize_a,
                                             np.zeros(self.adim),
                                             minimizer_kwargs={
                                                 'args': (poi, ),
                                                 'method': "L-BFGS-B",
                                                 'bounds': bounds
                                             }).fun, X_pois)))

    def _optimal_choice_da(self, X_pois: np.ndarray):
        bounds = np.stack((-1 * np.ones(self.adim), np.ones(self.adim))). \
            T.reshape(self.adim, 2)
        return np.array(
            list(
                map(
                    lambda poi: dual_annealing(self._q_to_optimize_a,
                                               args=(poi, ),
                                               bounds=bounds).fun, X_pois)))

    def _optimal_choice_sample(self, X_pois: np.ndarray):
        """
        Just do it by sampling 10.000 times.
        """
        n = 10000
        choices = list()
        t = datetime.now()  # debug only
        i = 0  # debug only
        for x in X_pois:
            t_ = datetime.now()  # debug only
            if (t_ - t).total_seconds() > 10:  # debug only
                print(f"Sampled {i}/{X_pois.shape[0]}")  # debug only
            # TODO Add hyper parameter for “10000” (although it doesn't really matter)
            A = self._random_sample.uniform(-1, 1, (n, self.adim))
            choices.append(max(map(lambda a: self.q(np.append(x, a)), A)))
            i += 1  # debug only

        return np.array(choices)
