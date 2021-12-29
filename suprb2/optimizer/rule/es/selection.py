from abc import ABCMeta, abstractmethod

import numpy as np

from suprb2.base import BaseComponent
from suprb2.rule import Rule
from suprb2.utils import RandomState


class RuleSelection(BaseComponent, metaclass=ABCMeta):
    """Decides on the best rule of an iteration."""

    @abstractmethod
    def __call__(self, rules: list[Rule], random_state: RandomState) -> Rule:
        pass


class Fittest(RuleSelection):
    """Take the rule with highest fitness."""

    def __call__(self, rules: list[Rule], random_state: RandomState) -> Rule:
        return max(rules, key=lambda child: child.fitness_)


class RouletteWheel(RuleSelection):
    """Selection probability is proportional to fitness."""

    def __call__(self, rules: list[Rule], random_state: RandomState) -> Rule:
        fitness = np.array([rule.fitness_ for rule in rules])
        fitness /= np.sum(fitness)

        return random_state.choice(rules, p=fitness)


class NondominatedSort(RuleSelection):
    """Choose a random rule from the pareto front."""

    def __call__(self, rules: list[Rule], random_state: RandomState) -> Rule:
        candidates: list[Rule] = [rules[0]]
        for rule in rules[1:]:
            to_be_added = False
            for can in candidates:

                if can.error_ < rule.error_ and can.volume_ > rule.volume_:
                    # classifier is dominated by this candidate and should not
                    # become a new candidate
                    to_be_added = False
                    break
                elif can.error_ > rule.error_ and can.volume_ < rule.volume_:
                    # classifier dominates candidate
                    candidates.remove(can)
                    to_be_added = True
                else:
                    to_be_added = True

            if to_be_added:
                candidates.append(rule)

        return random_state.choice(candidates)
