from abc import abstractmethod, ABCMeta
from typing import Optional

import numpy as np

from suprb2.base import BaseComponent
from suprb2.individual import Individual
from suprb2.rule import Rule
from suprb2.utils import RandomState


class RuleOriginGeneration(BaseComponent, metaclass=ABCMeta):
    """Determines a set of examples to initiate new rules around, i.e., the origins of new rules."""

    @abstractmethod
    def __call__(self, n_rules: int, X: np.ndarray, pool: list[Rule], elitist: Optional[Individual],
                 random_state: RandomState) -> np.ndarray:
        pass


class UniformOrigin(RuleOriginGeneration):
    """Sample origins uniformly in the input space."""

    def __call__(self, n_rules: int, X: np.ndarray, pool: list[Rule], elitist: Optional[Individual],
                 random_state: RandomState) -> np.ndarray:
        return random_state.uniform(np.min(X, axis=0), np.max(X, axis=0), size=(n_rules, X.shape[1]))


class SubgroupMatching(RuleOriginGeneration):
    """
    Bias the examples that were matched less than others by rules in the particular subgroup
    defined by `_subgroup()` to have a higher probability to be selected.
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
    Bias the examples that were matched less than others by rules in the pool to
    have a higher probability to be selected.
    """

    def _subgroup(self, pool: list[Rule], elitist: Optional[Individual]) -> list[Rule]:
        return pool


class ElitistMatching(SubgroupMatching):
    """
    Bias the examples that were matched less than others by rules in the subpopulation of
    the current elitist (the current elitist solution) to have a higher probability to be selected.
    """

    def _subgroup(self, pool: list[Rule], elitist: Optional[Individual]) -> list[Rule]:
        if elitist is not None:
            return elitist.subpopulation
        else:
            return pool
