from typing import Optional

import numpy as np

from suprb2.rule import Rule, RuleInit
from suprb2.rule.initialization import HalfnormInit
from .mutation import RuleMutation, Normal
from .selection import RuleSelection, Fittest
from .. import RuleAcceptance, RuleConstraint
from ..acceptance import Variance
from ..base import SingleElitistRuleGeneration
from ..constraint import CombinedConstraint, MinRange, Clip
from ..origin import PoolMatching, RuleOriginSampling


class ES1xLambda(SingleElitistRuleGeneration):
    """ The 1xLambda Evolutionary Strategy, where x is in {,+&}.

    Parameters
    ----------
    n_iter: int
        Iterations to evolve a rule.
    sampling: RuleOriginSampling
        The sampling process which decides on the next initial points or bounds.
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
                 lmbda: int = 20,
                 operator: str = ',',
                 sampling: RuleOriginSampling = PoolMatching(),
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
            sampling=sampling,
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

    def _optimize(self, X: np.ndarray, y: np.ndarray, initial_rule: Rule) -> Optional[Rule]:

        elitist = initial_rule

        # Main iteration
        for iteration in range(self.n_iter):
            # Generate, fit and evaluate lambda children
            children = [self.constraint(self.mutation(elitist, random_state=self.random_state_))
                            .fit(X, y) for _ in range(self.lmbda)]

            # Filter children that do not match any points
            children = list(filter(lambda rule: rule.is_fitted_ and rule.experience_ > 0, children))

            # Different operators
            if self.operator == '+':
                children.append(elitist)
                elitist = self.selection(children, self.random_state_)
            elif self.operator == ',':
                elitist = self.selection(children, self.random_state_)
            elif self.operator == '&':
                candidate = self.selection(children, self.random_state_)
                if candidate.fitness_ <= elitist.fitness_:
                    break
                else:
                    elitist = candidate

        return elitist
