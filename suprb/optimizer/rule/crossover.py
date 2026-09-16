from types import SimpleNamespace
import numpy as np
from suprb.rule import Rule
from suprb.optimizer.rule.generation_operator import GenerationOperator
from suprb.rule.matching import MatchingFunction
from suprb.utils import RandomState
from copy import deepcopy


class RuleCrossover(GenerationOperator):

    def __init__(self, matching_type: MatchingFunction = None, crossover_rate: float = 0.2):
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

    def gabil(self, A: Rule, B: Rule, random_state: RandomState) -> list[Rule]:
        """Swap whole bitstrings between the parents, one attribute at a time.

        The interval variant swaps a [lower, upper] pair per dimension. The
        categorical counterpart is the bitstring for one attribute, so the
        same mask applies and nothing inside a bitstring is touched: a
        rule's condition on one attribute either comes from A or from B,
        never a mixture of both.

        The result stays a list rather than becoming an array, since
        bitstrings for different attributes may have different lengths.
        """
        a = deepcopy(A)
        b = deepcopy(B)
        a_bits = deepcopy(A.match.bits)
        b_bits = deepcopy(B.match.bits)

        bool_mask = random_state.choice([False, True], size=(len(a_bits),))

        a.match.bits = [a_bits[i] if bool_mask[i] else b_bits[i] for i in range(len(bool_mask))]
        b.match.bits = [b_bits[i] if bool_mask[i] else a_bits[i] for i in range(len(bool_mask))]

        return [a, b]

    def mixed_types(self, A: Rule, B: Rule, random_state: RandomState) -> list[Rule]:
        """Cross both halves with the operator each already has.

        MixedTypes composes two matching functions, so crossover forwards
        to the method written for each: the interval half swaps bound
        pairs, the categorical half swaps whole bitstrings. Both halves are
        crossed with their own mask, so an attribute is inherited
        independently of the others, which is what the interval variant
        already does per dimension.

        The existing methods read `rule.match.bounds` and `rule.match.bits`
        directly, so each half is wrapped in a small object exposing it as
        `match`. Only the resulting arrays are taken from the wrappers; the
        children themselves are copies of the original rules, so their
        masks and observed categories are preserved.
        """
        a = deepcopy(A)
        b = deepcopy(B)

        reals = self.ordered_bound(
            SimpleNamespace(match=A.match.reals),
            SimpleNamespace(match=B.match.reals),
            random_state,
        )
        categoricals = self.gabil(
            SimpleNamespace(match=A.match.categoricals),
            SimpleNamespace(match=B.match.categoricals),
            random_state,
        )

        a.match.reals.bounds = reals[0].match.bounds
        b.match.reals.bounds = reals[1].match.bounds

        a.match.categoricals.bits = categoricals[0].match.bits
        b.match.categoricals.bits = categoricals[1].match.bits

        return [a, b]
