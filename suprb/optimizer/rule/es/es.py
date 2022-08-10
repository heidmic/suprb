import warnings
from collections import deque
from typing import Optional

import numpy as np

from suprb.rule import Rule, RuleInit
from suprb.rule.matching import OrderedBound, UnorderedBound, CentreSpread, MinPercentage
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
                 adaptive_sigma: bool = False,
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
        self.adaptive_sigma = adaptive_sigma
        self.lmbda = lmbda
        self.delay = delay
        self.operator = operator
        self.mutation = mutation
        self.selection = selection

    def _optimize(self, X: np.ndarray, y: np.ndarray, initial_rule: Rule, random_state: RandomState) -> Optional[Rule]:

        elitist = initial_rule

        elitists = deque(maxlen=self.delay)

        # Main iteration
        for iteration in range(self.n_iter):
            elitists.append(elitist)

            # Generate, fit and evaluate lambda children
            children = [self.constraint(self.mutation(elitist, random_state=random_state))
                            .fit(X, y) for _ in range(self.lmbda)]

            # Filter children that do not match any data samples
            valid_children = list(filter(lambda rule: rule.is_fitted_ and rule.experience_ > 0, children))

            if valid_children:
                children = valid_children
            else:
                warnings.warn("No valid children were generated during this iteration.", UserWarning)
                continue

            # For CSR only the spread mutation rate gets altered, for MPR both rate are altered
            if self.adaptive_sigma:
                matching_type = elitist.match.__class__.__name__
                fitter_children = sum([1 for child in children if child.fitness_ > elitist.fitness_])
                proportion = fitter_children / self.lmbda

                if proportion > 0.2:
                    if matching_type in ("OrderedBound", "UnorderedBound"):
                        self.mutation.sigma /= 0.85
                    elif matching_type == "CentreSpread":
                        self.mutation.sigma[1] /= 0.85
                    elif matching_type == "MinPercentage":
                        self.mutation.sigma[0] /= random_state.normal(loc=0.85, scale=0.01)
                        self.mutation.sigma[1] /= 0.85
                elif 0.05 <= proportion < 0.2:
                    if matching_type in ("OrderedBound", "UnorderedBound"):
                        self.mutation.sigma *= 0.85
                    elif matching_type == "CentreSpread":
                        self.mutation.sigma[1] *= 0.85
                    elif matching_type == "MinPercentage":
                        self.mutation.sigma[0] *= random_state.normal(loc=0.85, scale=0.01)
                        self.mutation.sigma[1] *= 0.85
                elif proportion < 0.05:
                    if matching_type in ("OrderedBound", "UnorderedBound"):
                        self.mutation.sigma *= 2
                    elif matching_type == "CentreSpread":
                        self.mutation.sigma[1] *= 2
                    elif matching_type == "MinPercentage":
                        self.mutation.sigma[0] *= random_state.normal(loc=2, scale=0.01)
                        self.mutation.sigma[1] *= 2

                self.mutation.sigma = np.clip(self.mutation.sigma, 0.001, 3)

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

        return elitist
