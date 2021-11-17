from abc import ABCMeta
from typing import Union

import numpy as np

from suprb2.optimizer import BaseOptimizer
from suprb2.rule import Rule
from . import RuleInit, RuleFitness, RuleAcceptance, RuleConstraint


class RuleGeneration(BaseOptimizer, metaclass=ABCMeta):
    """ Base class of different methods to generate `Rule`s.

    Parameters
    ----------
    n_iter: int
        Iterations to evolve a rule.
    start: Rule
        The elitist this optimizer starts on.
    mean: np.ndarray
        Mean to generate a rule from. The parameter `start` has priority.
    init: RuleInit
    fitness: RuleFitness
    acceptance: RuleAcceptance
    constraint: RuleConstraint
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
    n_jobs: int
        The number of threads / processes the optimization uses.
    """

    def __init__(self,
                 n_iter: int,
                 start: Rule,
                 mean: np.ndarray,
                 init: RuleInit,
                 fitness: RuleFitness,
                 acceptance: RuleAcceptance,
                 constraint: RuleConstraint,
                 random_state: int,
                 n_jobs: int,
                 ):
        super().__init__(random_state=random_state, n_jobs=n_jobs)

        self.n_iter = n_iter
        self.start = start
        self.mean = mean
        self.init = init
        self.fitness = fitness
        self.acceptance = acceptance
        self.constraint = constraint

    def optimize(self, X: np.ndarray, y: np.ndarray) -> Union[Rule, list[Rule], None]:
        pass


class SingleSolutionBasedRuleGeneration(RuleGeneration, metaclass=ABCMeta):
    """Base class of single-solution-based optimizers to generate `Rule`s."""

    elitist_: Rule

    def _init_elitist(self, X, y):
        """Generate starting rule, if not provided."""
        if self.start is None:
            self.elitist_ = self.init(mean=self.mean, random_state=self.random_state_)
            self.constraint(self.elitist_).fit(X, y, self.fitness)
        else:
            self.elitist_ = self.constraint(self.start).fit(X, y, self.fitness)

    def elitist(self):
        return self.elitist_

    def _reset(self):
        super()._reset()
        if hasattr(self, 'elitist_'):
            del self.elitist_
