from pymoo.core.mutation import Mutation
import numpy as np
from suprb.optimizer.rule.mutation import RuleMutation
from suprb.rule.base import Rule
from suprb.optimizer.rule.constraint import RuleConstraint

class PymooRuleMutation(Mutation):
    """
    Wrapper class for SupRB's mutation to work with pymoo.
    Deprecated.
    """
    def __init__(self, initial_rule: Rule, mutation_operator: RuleMutation, constraint: RuleConstraint, X_train, y_train, random_state):
        super().__init__()
        self.initial_rule = initial_rule
        self.mutation_operator = mutation_operator
        self.constraint = constraint
        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state

    # def _do(self, problem, X, **kwargs):
    #     def mutate(params):
    #         r = self.initial_rule.clone().set_param_vector(params)
    #         r = self.mutation_operator(r, self.random_state)
    #         r = self.constraint(r)
    #         return r.get_param_vector()
        
    #     mutated_rules = [mutate(p) for p in X]
    #     return np.array(mutated_rules)

    def _do(self, problem, X, **kwargs):
        mutated_population = []

        for params in X:
            rule = self.initial_rule.clone()
            rule.set_param_vector(params)

            mutated_rule = self.constraint(self.mutation_operator(rule, self.random_state)).fit(self.X_train, self.y_train)

            if mutated_rule.is_fitted_ and mutated_rule.experience_ > 0:
                mutated_population.append(mutated_rule.get_param_vector())
            else:
                mutated_population.append(params)

        return np.array(mutated_population)