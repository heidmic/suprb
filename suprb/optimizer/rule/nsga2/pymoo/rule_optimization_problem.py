import numpy as np

from suprb.rule import Rule

from suprb.optimizer.rule import RuleConstraint

from pymoo.core.problem import Problem


class RuleOptimizationProblem(Problem):
    """
    Wrapper class for SupRB's optimization problem to work with pymoo.
    Deprecated.
    """
    def __init__(self, rule: Rule, X: np.ndarray, y: np.ndarray, constraint: RuleConstraint):
        self.rule_template = rule
        self.X = X
        self.y = y
        self.param_len = len(rule.get_param_vector())
        self.constraint = constraint

        xl = np.full(self.param_len, -1.0)
        xu = np.full(self.param_len, 1.0)

        super().__init__(
            n_var=self.param_len,
            n_obj=2,
            n_constr=0,
            xl=xl,
            xu=xu,
            elementwise_evaluation=False
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for x in X:
            rule = self.rule_template.clone().set_param_vector(x)

            rule = rule.fit(self.X, self.y)

            if not rule.is_fitted_ or rule.experience_ == 0:
                F.append([np.inf, np.inf])
            else:
                rule = self.constraint(rule)
                F.append([rule.error_, -rule.volume_])

        out["F"] = np.array(F)
