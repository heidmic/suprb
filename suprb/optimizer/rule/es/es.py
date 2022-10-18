from copy import deepcopy
import warnings
from collections import deque
from typing import Optional

import numpy as np
import warnings

from suprb.rule import Rule, RuleInit
from suprb.rule.initialization import MeanInit
from suprb.utils import RandomState
from ..mutation import RuleMutation, HalfnormIncrease
from ..selection import RuleSelection, Fittest
from .. import RuleAcceptance, RuleConstraint
from ..acceptance import Variance
from ..base import ParallelSingleRuleGeneration
from ..constraint import CombinedConstraint, MinRange, Clip
from ..origin import Matching, RuleOriginGeneration


class ES1xLambda(ParallelSingleRuleGeneration):
    """ The 1xLambda Evolutionary Strategy, where x is in {,+&}.

    Parameters
    ----------
    n_iter: int
        Iterations to evolve a rule.
    lmbda: int
        Children to generate in every iteration.
    operator: str
        Can be one of ',', '+' or '&'.
        ',' replaces the elitist in every generation.
        '+' may keep the elitist.
        '&' behaves similar to '+' and ends the optimization process, if no improvement is found in a generation.
    delay: int
        Only relevant if operator is '&'. Controls the number of elitists which need to be worse before stopping.
    origin_generation: RuleOriginGeneration
        The selection process which decides on the next initial points.
    init: RuleInit
    mutation: RuleMutation
        Default is HalfnormIncrease(sigma=1.22)
    selection: RuleSelection
    acceptance: RuleAcceptance
    constraint: RuleConstraint
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
    n_jobs: int
        The number of threads / processes the optimization uses. Currently not used for this optimizer.
    """

    def __init__(self,
                 n_iter: int = 10_000,
                 lmbda: int = 20,
                 operator: str = '&',
                 delay: int = 146,
                 origin_generation: RuleOriginGeneration = Matching(),
                 init: RuleInit = MeanInit(),
                 mutation: RuleMutation = HalfnormIncrease(sigma=1.22),
                 selection: RuleSelection = Fittest(),
                 acceptance: RuleAcceptance = Variance(),
                 constraint: RuleConstraint = CombinedConstraint(MinRange(), Clip()),
                 random_state: int = None,
                 n_jobs: int = 1,
                 ):
        super().__init__(
            n_iter=n_iter,
            origin_generation=origin_generation,
            init=init,
            acceptance=acceptance,
            constraint=constraint,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.lmbda = lmbda
        self.delay = delay
        self.operator = operator
        self.mutation = mutation
        self.selection = selection

        if self.delay < 2:
            warnings.warn("Delay too low! Volume of rules will not increase! A delay of one does not allow mutation to happen")

        if self.operator == '&':
            assert self.delay < self.n_iter, f"n_iter {self.n_iter} must be " \
                                             f"greater than delay {self.delay}"
        if self.operator == ',' and isinstance(self.mutation, HalfnormIncrease):
            warnings.warn("',' operator and HalfnormIncrease mutation lead to collapsing populations")

    def _optimize(self, X: np.ndarray, y: np.ndarray, initial_rule: Rule, random_state: RandomState) -> Optional[Rule]:
        elitist = initial_rule
        elitists = deque(maxlen=self.delay)
        mutation = deepcopy(self.mutation)

        # Main iteration
        for iteration in range(self.n_iter):
            elitists.append(elitist)

            # Generate, fit and evaluate lambda children
            children = [self.constraint(mutation(elitist, random_state=random_state))
                            .fit(X, y) for _ in range(self.lmbda)]

            # Filter children that do not match any data samples
            valid_children = list(filter(lambda rule: rule.is_fitted_ and rule.experience_ > 0, children))

            if valid_children:
                children = valid_children
            else:
                warnings.warn("No valid children were generated during this iteration.", UserWarning)
                continue

            # Different operators for replacement
            # 'selection' returns a list of rules. Either unordered or
            # descending, we thus take the first element for our new parent
            if self.operator == '+':
                children.append(elitist)
                elitist = self.selection(children, random_state=random_state)[0]
            elif self.operator in (',', '&'):
                elitist = self.selection(children, random_state=random_state)[0]
            if self.operator == '&':
                if len(elitists) == self.delay and all([e.fitness_ <= elitists[0].fitness_ for e in elitists]):
                    elitist = elitists[0]
                    break

            if getattr(mutation, "adapt", None):
                mutation.adapt(elitist.fitness_)

        return elitist
