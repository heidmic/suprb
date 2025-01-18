import numpy as np

from suprb.rule import Rule
from suprb.rule.initialization import RuleInit, HalfnormInit
from suprb.utils import RandomState, check_random_state
from .. import RuleAcceptance, RuleConstraint
from ..acceptance import Variance
from ..base import RuleDiscovery
from ..constraint import CombinedConstraint, MinRange, Clip
from ..origin import RuleOriginGeneration, SquaredError
from ..selection import RuleSelection, Fittest


class RandomSearch(RuleDiscovery):
    """ RandomSearch Algorithm

    Parameters
    ----------
    n_iter: int
        Has no effect
    rules_generated: float
        Number of random rules generated per rule added to the pool
    origin_generation: RuleOriginGeneration
        The selection process which decides on the next initial points.
    init: RuleInit
        A method to init rules. The init must always match at least one
        example but ideally should already match more than one,
        e.g. HalfnormInit, whereas NormalInit would not work consistently.

    selection: RuleSelection
    acceptance: RuleAcceptance
    constraint: RuleConstraint

    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
    n_jobs: int
        The number of threads / processes the optimization uses. Currently not used for this optimizer.
    """

    last_iter_inner: bool

    def __init__(self,
                 n_iter: int = 1,
                 rules_generated: int = 7,

                 origin_generation: RuleOriginGeneration = SquaredError(),
                 init: RuleInit = HalfnormInit(),

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

        self.selection = selection
        self.rules_generated = rules_generated

    def optimize(self, X: np.ndarray, y: np.ndarray, n_rules: int = 1) -> list[Rule]:
        """
        Generates n_rules * self.rules_generated rules randomly and
        returns the best n_rules candidates according to the selection
        criterion. Generation and selection occur independently for each
        returned rule

        Return: A list of filtered Rules.
        """
        self.random_state_ = check_random_state(self.random_state)

        rules_out = []
        for _ in range(n_rules):
            origins = self.origin_generation(n_rules=self.rules_generated, X=X,
                                             y=y, pool=self.pool_,
                                             elitist=self.elitist_,
                                             random_state=self.random_state_)

            rules = []
            for origin in origins:
                rule = self.init(mean=origin, random_state=self.random_state_)
                rules.append(self.constraint(rule).fit(X, y))

            rules_out.extend(self.selection(rules, random_state=self.random_state_, size=1))

        return self._filter_invalid_rules(X=X, y=y, rules=rules_out)
