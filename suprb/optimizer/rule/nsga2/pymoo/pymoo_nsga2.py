import numpy as np
from typing import Optional

from suprb.rule import Rule, RuleInit
from suprb.rule.initialization import MeanInit
from suprb.optimizer.rule.nsga2.pymoo.rule_optimization_problem import RuleOptimizationProblem
from suprb.optimizer.rule.nsga2.pymoo.pymoo_mutation import PymooRuleMutation
from suprb.utils import RandomState
from suprb.optimizer.rule.base import ParallelSingleRuleDiscovery
from suprb.optimizer.rule.origin import SquaredError, RuleOriginGeneration
from suprb.optimizer.rule.mutation import RuleMutation, HalfnormIncrease
from suprb.optimizer.rule.constraint import CombinedConstraint, MinRange, Clip
from suprb.optimizer.rule.acceptance import Variance
from suprb.optimizer.rule import RuleAcceptance, RuleConstraint


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

import cProfile
import pstats



class PymooNSGA2(ParallelSingleRuleDiscovery):
    """
    Adapted from: A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II by Kalyanmoy Deb et al.
    In the original plan for the MOO-RD was to use pymoo's precompiled algorithms.
    This was slower in the end as repeated vectorization of the rule objects was necessary.
    Deprecated.
    """
    def __init__(
            self,
            n_iter: int = 100,
            mu: int = 20,
            lmbda: int = 40,
            origin_generation: RuleOriginGeneration = SquaredError(),
            init: RuleInit = MeanInit(),
            mutation: RuleMutation = HalfnormIncrease(sigma=1.22),
            constraint: RuleConstraint = CombinedConstraint(MinRange(), Clip()),
            acceptance: RuleAcceptance = Variance(),
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
        self.mu = mu
        self.lmbda = lmbda
        self.mutation = mutation
        self.constraint = constraint

    def _optimize(
            self,
            X: np.ndarray,
            y: np.ndarray,
            initial_rule: Rule,
            random_state: RandomState,
    ) -> Optional[Rule]:
        rule_template = initial_rule.clone().fit(X, y)

        problem = RuleOptimizationProblem(
            rule_template,
            X,
            y, 
            constraint=self.constraint,
        )

        mutation = PymooRuleMutation(
            rule_template,
            mutation_operator=self.mutation,
            constraint=self.constraint,
            X_train=X,
            y_train=y,
            random_state=random_state,
        )

        algorithm = NSGA2(
            pop_size=self.lmbda,
            mutation=mutation,
            #mutation=PM(eta=50, prob=0.8),
            #crossover=SBX(eta=, prob=0.9),
            eliminate_duplicates=False,
        )

        termination = get_termination(
            "n_gen",
            self.n_iter
        )

         # ─────────────── START PROFILING ───────────────
        profiler = cProfile.Profile()
        profiler.enable()

        result = minimize(
            problem,
            algorithm,
            termination,
            seed=int(random_state.integers(0, 2**32 - 1)),
            verbose=False,
        )

        profiler.disable()
        # Dump the top 20 functions by cumulative time
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats(20)
        # ─────────────── END PROFILING ─────────────────

        population = result.algorithm.pop

        F = population.get("F")
        Xf = population.get("X")
        rank = population.get("rank")

    
        pf_idx = np.where(rank == 0)[0]
        if pf_idx.size == 0:
            return None

        best_idx = pf_idx[np.argmin(F[pf_idx, 0])]

        best_vec  = Xf[best_idx]
        best_rule = rule_template.clone().set_param_vector(best_vec)
        best_rule = best_rule.fit(X, y)

        return best_rule if best_rule.is_fitted_ and best_rule.experience_ > 0 else None
