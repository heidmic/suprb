import numpy as np
from suprb.rule import Rule
from suprb.optimizer.rule.generation_operator import GenerationOperator
from suprb.rule.matching import MatchingFunction
from suprb.utils import RandomState
from copy import deepcopy


class RuleCrossover(GenerationOperator):

    def __init__(self,
                 matching_type: MatchingFunction = None,
                 crossover_rate: float = 0.2):
        super().__init__(matching_type=matching_type)
        self.crossover_rate = crossover_rate

    def __call__(self, A: Rule, B: Rule, random_state: RandomState) -> list[Rule]:
        if random_state.random() < self.crossover_rate:
            return self.execute(A=A, B=B, random_state=random_state)

        return [A, B]

    def execute(self, A: Rule, B: Rule, random_state: RandomState):
        pass


class UniformCrossover(RuleCrossover):
    """Decide for every bound tuple with uniform probability if the bound tuple in rule A or B is used."""

    def ordered_bound(self, A: Rule, B: Rule, random_state: RandomState) -> list[Rule]:
        a = deepcopy(A)
        b = deepcopy(B)
        a_bounds = deepcopy(A.match.bounds)
        b_bounds = deepcopy(B.match.bounds)

        bool_mask = random_state.choice([False, True], size=(len(a_bounds),))

        a.match.bounds = np.array([a_bounds[i] if bool_mask[i] else b_bounds[i] for i in range(len(bool_mask))])
        b.match.bounds = np.array([b_bounds[i] if bool_mask[i] else a_bounds[i] for i in range(len(bool_mask))])

        return [a, b]

    def unordered_bound(self, rule: Rule, random_state: RandomState):
        raise NotImplementedError

    def center_spread(self, rule: Rule, random_state: RandomState):
        raise NotImplementedError

    def min_percentage(self, rule: Rule, random_state: RandomState):
        raise NotImplementedError
