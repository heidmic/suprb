from typing import Optional

import numpy as np

from suprb2.rule import Rule, RuleInit
from suprb2.rule.initialization import HalfnormInit
from suprb2.utils import check_random_state
from .mutation import RuleMutation, Normal
from .selection import RuleSelection, Fittest
from .. import RuleAcceptance, RuleConstraint
from ..acceptance import Variance
from ..base import SingleSolutionBasedRuleGeneration
from ..constraint import CombinedConstraint, MinRange, Clip


class ES1xLambda(SingleSolutionBasedRuleGeneration):
    """ The 1xLambda Evolutionary Strategy, where x is in {,+&}.

    Parameters
    ----------
    n_iter: int
        Iterations to evolve a rule.
    start: Rule
        The elitist this optimizer starts on.
    mean: np.ndarray
        Mean to generate a rule from. The parameter `start` has priority.
    lmbda: int
        Children to generate in every iteration.
    operator: str
        Can be one of ',', '+' or '&'.
        ',' replaces the elitist in every generation.
        '+' may keep the elitist.
        '&' behaves similar to '+' and ends the optimization process, if no improvement is found in a generation.
    init: RuleInit
    mutation: RuleMutation
    selection: RuleSelection
    acceptance: RuleAcceptance
    constraint: RuleConstraint
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
    n_jobs: int
        The number of threads / processes the optimization uses. Currently not used for this optimizer.
    """

    def __init__(self,
                 n_iter: int = 10,
                 start: Rule = None,
                 mean: np.ndarray = None,
                 lmbda: int = 20,
                 operator: str = ',',
                 init: RuleInit = HalfnormInit(),
                 mutation: RuleMutation = Normal(),
                 selection: RuleSelection = Fittest(),
                 acceptance: RuleAcceptance = Variance(),
                 constraint: RuleConstraint = CombinedConstraint(MinRange(), Clip()),
                 random_state: int = None,
                 n_jobs: int = 1,
                 ):
        super().__init__(
            n_iter=n_iter,
            start=start,
            mean=mean,
            init=init,
            acceptance=acceptance,
            constraint=constraint,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.lmbda = lmbda
        self.operator = operator
        self.mutation = mutation
        self.selection = selection

    def optimize(self, X: np.ndarray, y: np.ndarray) -> Optional[Rule]:
        self.random_state_ = check_random_state(self.random_state)

        self._init_elitist(X, y)

        # Main iteration
        for iteration in range(self.n_iter):
            # Generate, fit and evaluate lambda children
            children = [self.constraint(self.mutation(self.elitist_, random_state=self.random_state_))
                            .fit(X, y) for _ in range(self.lmbda)]
            # Filter children that do not match any points
            filtered = list(filter(lambda rule: rule.is_fitted_ and rule.experience_ > 0, children))

            def candidate(population: list[Rule]):
                return self.selection(population, self.random_state_) if population else self.elitist_

            # Different operators
            if self.operator == '+':
                filtered.append(self.elitist_)
                self.elitist_ = candidate(filtered)
            elif self.operator == ',':
                self.elitist_ = candidate(filtered)
            elif self.operator == '&':
                candidate = candidate(filtered)
                if candidate.fitness_ <= self.elitist_.fitness_:
                    break
                else:
                    self.elitist_ = candidate

        # Rules under a threshold error are appended to the pool
        if self.acceptance(self.elitist_, X, y):
            return self.elitist_
        else:
            return None
