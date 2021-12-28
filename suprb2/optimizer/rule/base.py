from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np
from joblib import Parallel, delayed

from suprb2.individual import Individual
from suprb2.optimizer import BaseOptimizer
from suprb2.rule import Rule, RuleInit
from .acceptance import RuleAcceptance
from .constraint import RuleConstraint
from .origin import RuleOriginGeneration
from ...utils import check_random_state, RandomState, spawn_random_states


class RuleGeneration(BaseOptimizer, metaclass=ABCMeta):
    """ Base class of different methods to generate `Rule`s.

    Parameters
    ----------
    n_iter: int
        Iterations to evolve a rule.
    origin_generation: RuleOriginGeneration
        The sampling process which decides on the next initial points or bounds.
    init: RuleInit
    acceptance: RuleAcceptance
    constraint: RuleConstraint
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
    n_jobs: int
        The number of threads / processes the optimization uses.
    """

    pool_: list[Rule]
    elitist_: Individual

    def __init__(self,
                 n_iter: int,
                 origin_generation: RuleOriginGeneration,
                 init: RuleInit,
                 acceptance: RuleAcceptance,
                 constraint: RuleConstraint,
                 random_state: int,
                 n_jobs: int,
                 ):
        super().__init__(random_state=random_state, n_jobs=n_jobs)

        self.n_iter = n_iter
        self.origin_generation = origin_generation
        self.init = init
        self.acceptance = acceptance
        self.constraint = constraint

    def _filter_invalid_rules(self, X: np.ndarray, y: np.ndarray, rules: list[Rule]) -> list[Rule]:
        return list(filter(lambda rule: rule is not None and self.acceptance(rule=rule, X=X, y=y), rules))

    @abstractmethod
    def optimize(self, X: np.ndarray, y: np.ndarray, n_rules: int = 1) -> list[Rule]:
        pass


class ParallelSingleRuleGeneration(RuleGeneration, metaclass=ABCMeta):
    """
    Implements basic functionality to generate several `Rule`s in parallel.
    The optimization process is assumed to generate exactly one rule for every origin data sample.
    """

    def optimize(self, X: np.ndarray, y: np.ndarray, n_rules: int = 1) -> list[Rule]:
        self.random_state_ = check_random_state(self.random_state)
        random_states = spawn_random_states(self.random_state_, n=n_rules)

        origins = self.origin_generation(n_rules=n_rules, X=X, pool=self.pool_, elitist=self.elitist_,
                                         random_state=self.random_state_)

        initial_rules = []
        for origin in origins:
            initial_rule = self.init(mean=origin, random_state=self.random_state_)
            initial_rules.append(self.constraint(initial_rule).fit(X, y))

        with Parallel(n_jobs=self.n_jobs) as parallel:
            rules = parallel(delayed(self._optimize)(X=X, y=y, initial_rule=initial_rule, random_state=random_state)
                             for initial_rule, random_state in zip(initial_rules, random_states))

        return self._filter_invalid_rules(X=X, y=y, rules=rules)

    @abstractmethod
    def _optimize(self, X: np.ndarray, y: np.ndarray, initial_rule: Rule, random_state: RandomState) -> Optional[Rule]:
        pass
