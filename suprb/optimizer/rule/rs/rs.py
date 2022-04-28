import numpy as np

from suprb.rule import Rule
from suprb.rule.initialization import HalfnormInit
from suprb.utils import RandomState, check_random_state
from .. import RuleAcceptance, RuleConstraint
from ..acceptance import Variance
from ..base import RuleGeneration
from ..constraint import CombinedConstraint, MinRange, Clip
from ..origin import RuleOriginGeneration, UniformSamplesOrigin
from ..selection import RuleSelection, Fittest


class RandomSearch(RuleGeneration):
    """ RandomSearch Algorithm

        Parameters
        ----------
        n_iter: int
            Iterations to evolve rules.
        rules_generated_ratio: float
            Number of rules generated
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
                 rules_generated_ratio: float = 7,

                 origin_generation: RuleOriginGeneration = Matching(),
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
        self.rules_generated_ratio = rules_generated_ratio

    def optimize(self, X: np.ndarray, y: np.ndarray, n_rules: int = 1) -> list[Rule]:
        """
        Generates n_rules * rules_generated_ratio rules randomly and
        returns the best n_rules candidates according to the selection
        criterion

        Return: A list of filtered Rules.
        """
        # self._validate_params(n_rules)

        self.random_state_ = check_random_state(self.random_state)

        n_rules_generated = np.round(n_rules * self.rules_generated_ratio)

        origins = self.origin_generation(n_rules=n_rules_generated, X=X, y=y,
                                         pool=self.pool_, elitist=self.elitist_,
                                         random_state=self.random_state_)

        rules = []
        for origin in origins:
            rule = self.init(mean=origin, random_state=self.random_state_)
            rules.append(self.constraint(rule).fit(X, y))

        rules = self.selection(rules, size=n_rules)

        return self._filter_invalid_rules(X=X, y=y, rules=rules)
