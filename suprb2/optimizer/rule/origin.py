from abc import abstractmethod
from typing import Optional

import numpy as np

from suprb2.base import BaseComponent
from suprb2.individual import Individual
from suprb2.rule import Rule
from suprb2.utils import RandomState


class RuleOriginSampling(BaseComponent):
    """Samples neglected single points from the current state of the model to generate new rules from."""

    @abstractmethod
    def __call__(self, n_rules: int, X: np.ndarray, pool: list[Rule], elitist: Optional[Individual],
                 random_state: RandomState) -> np.ndarray:
        pass


class SubgroupMatching(RuleOriginSampling):
    """
    Bias the indices so that input values that were matched less than others by rules in the particular subgroup
    defined by `_subgroup()` have a higher probability to be selected.
    """

    def __call__(self, n_rules: int, X: np.ndarray, pool: list[Rule], elitist: Optional[Individual],
                 random_state: RandomState) -> np.ndarray:

        subgroup = self._subgroup(pool=pool, elitist=elitist)

        if subgroup:
            counts = np.count_nonzero(np.stack([rule.match_ for rule in subgroup], axis=0) == 0, axis=0)
            counts_sum = np.sum(counts)
            # If all input values are matched by every rule, no bias is needed
            probabilities = counts / counts_sum if counts_sum != 0 else None
        else:
            # No bias needed when no rule exists
            probabilities = None

        indices = random_state.choice(np.arange(len(X)), n_rules, p=probabilities)
        return X[indices]

    @abstractmethod
    def _subgroup(self, pool: list[Rule], elitist: Optional[Individual]) -> list[Rule]:
        pass


class PoolMatching(SubgroupMatching):
    """
    Bias the indices so that input values that were matched less than others by rules in the pool
    have a higher probability to be selected.
    """

    def _subgroup(self, pool: list[Rule], elitist: Optional[Individual]) -> list[Rule]:
        return pool


class ElitistMatching(SubgroupMatching):
    """
    Bias the indices so that input values that were matched less than others by rules in the subpopulation of
    the current elitist have a higher probability to be selected.
    """

    def _subgroup(self, pool: list[Rule], elitist: Optional[Individual]) -> list[Rule]:
        if elitist is not None:
            return elitist.subpopulation
        else:
            return pool
