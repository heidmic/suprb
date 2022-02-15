from abc import abstractmethod, ABCMeta
from typing import Optional

import numpy as np

from suprb2.base import BaseComponent
from suprb2.solution import Solution
from suprb2.rule import Rule
from suprb2.utils import RandomState


class RuleOriginGeneration(BaseComponent, metaclass=ABCMeta):
    """Determines a set of examples to initiate new rules around, i.e., the origins of new rules."""

    @abstractmethod
    def __call__(self, n_rules: int, X: np.ndarray, y: np.ndarray, pool: list[Rule], elitist: Optional[Solution],
                 random_state: RandomState) -> np.ndarray:
        pass


class UniformInputOrigin(RuleOriginGeneration):
    """Sample origins uniformly in the input space."""

    def __call__(self, n_rules: int, X: np.ndarray, random_state: RandomState, **kwargs) -> np.ndarray:
        return random_state.uniform(np.min(X, axis=0), np.max(X, axis=0), size=(n_rules, X.shape[1]))


class UniformSamplesOrigin(RuleOriginGeneration):
    """Sample origins uniformly in the sample space."""

    def __call__(self, n_rules: int, X: np.ndarray, random_state: RandomState, **kwargs) -> np.ndarray:
        return random_state.choice(X, axis=0, size=n_rules)


class RouletteWheelOrigin(RuleOriginGeneration):
    """
    Sample origins with weights calculated from rules in the pool.
    If `use_elitist` is set, the matching is only calculated on
     the subpopulation of the current elitist (the current elitist solution).
    """

    def __init__(self, use_elitist: bool = True):
        self.use_elitist = use_elitist

    def __call__(self, n_rules: int, X: np.ndarray, pool: list[Rule], elitist: Optional[Solution],
                 random_state: RandomState, **kwargs) -> np.ndarray:

        subgroup = elitist.subpopulation if elitist is not None and self.use_elitist else pool

        if subgroup:
            weights = self._calculate_weights(subgroup=subgroup, X=X, elitist=elitist, random_state=random_state,
                                              **kwargs)
            weights_sum = np.sum(weights)
            # If all weights are zero, no bias is needed
            probabilities = weights / weights_sum if weights_sum != 0 else None
        else:
            # No bias needed when no rule exists
            probabilities = None

        indices = random_state.choice(np.arange(len(X)), n_rules, p=probabilities)
        return X[indices]

    @abstractmethod
    def _calculate_weights(self, subgroup: list[Rule], **kwargs) -> np.ndarray:
        pass


class Matching(RouletteWheelOrigin):
    """Bias the examples that were matched less than others by rules to have a higher probability to be selected."""

    def _calculate_weights(self, subgroup: list[Rule], **kwargs) -> np.ndarray:
        return np.count_nonzero(np.stack([rule.match_ for rule in subgroup], axis=0) == 0, axis=0)


class SquaredError(RouletteWheelOrigin):
    """Bias the examples that have higher squared error on rules to have a higher probability to be selected."""

    def _calculate_weights(self, X: np.ndarray = None, y: np.ndarray = None, elitist: Solution = None,
                           **kwargs) -> np.ndarray:

        if self.use_elitist:
            pred = elitist.predict(X)
        else:
            pred = elitist.clone(genome=np.ones(len(elitist.pool))).predict(X)

        return (pred - y) ** 2
