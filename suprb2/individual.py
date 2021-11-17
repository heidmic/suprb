from __future__ import annotations

import itertools
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error

from .base import BaseComponent
from .optimizer import Solution
from .rule import Rule


class MixingModel(BaseComponent, metaclass=ABCMeta):
    """Performs mixing of local `Rule`s to obtain a complete prediction of the input space."""

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, X: np.ndarray, subpopulation: list[Rule], cache=False) -> np.ndarray:
        pass


class ErrorExperienceHeuristic(MixingModel):
    """
    Performs mixing similar to the Inverse Variance Heuristic from
    https://researchportal.bath.ac.uk/en/studentTheses/learning-classifier-systems-from-first-principles-a-probabilistic,
    but using (error / experience) as a mixing function.
    """

    def __call__(self, X: np.ndarray, subpopulation: list[Rule], cache=False) -> np.ndarray:
        input_size = X.shape[0]

        # No need to perform any calculation if no rule was selected.
        if not subpopulation:
            return np.zeros(X.shape[0])

        # Get errors and experience of all rules in subpopulation
        experiences = np.array([rule.experience_ for rule in subpopulation])
        errors = np.array([rule.error_ for rule in subpopulation])

        local_pred = np.zeros((len(subpopulation), input_size))

        if cache:
            # Use the precalculated matches and predictions from fit()
            matches = [rule.match_ for rule in subpopulation]
            for i, rule in enumerate(subpopulation):
                local_pred[i][matches[i]] = rule.pred_
        else:
            # Generate all data new
            matches = [rule.matched_data(X) for rule in subpopulation]
            for i, rule in enumerate(subpopulation):
                if not matches[i].any():
                    continue
                local_pred[i][matches[i]] = rule.predict(X[matches[i]])

        taus = (1 / errors) * experiences

        # Stack all local predictions and sum them weighted with tau
        pred = np.sum(local_pred * taus[:, None], axis=0)

        # Sum all taus
        local_taus = np.zeros((len(subpopulation), input_size))
        for i in range(len(subpopulation)):
            local_taus[i][matches[i]] = taus[i]

        tau_sum = np.sum(local_taus, axis=0)
        tau_sum[tau_sum == 0] = 1  # Needed

        # Normalize
        out = pred / tau_sum
        return out


class Individual(Solution):
    """Individual that mixes a subpopulation of rules with MoE."""

    input_size_: int
    complexity_: int

    def __init__(self, genome: np.ndarray, pool: list[Rule], mixture: MixingModel):
        self.genome = genome
        self.pool = pool
        self.mixture = mixture

    def fit(self, X: np.ndarray, y: np.ndarray, fitness) -> Individual:
        pred = self.predict(X, cache=True)
        self.error_ = max(mean_squared_error(y, pred), 1e-4)
        self.input_size_ = self.genome.shape[0]
        self.complexity_ = np.sum(self.genome).item()  # equivalent to np.count_nonzero, but possibly faster
        self.fitness_ = fitness(self)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray, cache=False) -> np.ndarray:
        """
        This function performs MoE of the rules in the subpopulation.

        The attribute `cache` decides if the prediction should use the cached data from the rules, which
        includes rule error, fitness, predictions and the binary match string.
        It is True while fitting, because the data is identical there and caching saves a good amount of time.
        For predictions after fitting, it is false, because all data needs to be recalculated from scratch.
        """

        return self.mixture(X=X, subpopulation=self.subpopulation, cache=cache)

    @property
    def subpopulation(self) -> list[Rule]:
        """Get all rules in the subpopulation."""
        assert (len(self.genome) == len(self.pool))
        return list(itertools.compress(self.pool, self.genome))

    def _more_str_attributes(self) -> dict:
        return {'complexity': self.complexity_}
