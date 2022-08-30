from abc import ABCMeta, abstractmethod

import numpy as np

from suprb.base import BaseComponent
from suprb.rule import Rule
from suprb.utils import RandomState


class RuleSelection(BaseComponent, metaclass=ABCMeta):
    """Decides on the best rule of an iteration."""

    @abstractmethod
    def __call__(self, rules: list[Rule], random_state: RandomState, size: int = 1) -> list[Rule]:
        pass


class Fittest(RuleSelection):
    """Take the rule with highest fitness."""

    def __call__(self, rules: list[Rule], random_state: RandomState, size: int = 1) -> list[Rule]:
        rules = sorted(rules, key=lambda child: child.fitness_, reverse=True)
        return rules[:size]


class RouletteWheel(RuleSelection):
    """Selection probability is proportional to fitness."""

    def __call__(self, rules: list[Rule], random_state: RandomState, size: int = 1) -> list[Rule]:
        try:
            fitnesses = np.array([rule.fitness_ for rule in rules])
            weights = fitnesses / np.sum(fitnesses)

            return random_state.choice(rules, p=weights, size=size)
        except ValueError:  # occurs if one or more rules did not match anything
            rules = [rule for rule in rules if rule.fitness_ != -np.inf]
            if rules == []:
                # if all rules didn't match anything we select randomly
                return random_state.choice(rules, size=size)
            else:
                # if some rules matched something we perform roulette wheel
                # only among those rules
                fitnesses = np.array([rule.fitness_ for rule in rules])
                weights = fitnesses / np.sum(fitnesses)

                return random_state.choice(rules, p=weights, size=size)


class NondominatedSort(RuleSelection):
    """Choose a random rule from the pareto front."""

    def __call__(self, rules: list[Rule], random_state: RandomState, size: int = 1) -> list[Rule]:
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

        return random_state.choice(candidates, size=size)


class Random(RuleSelection):

    def __call__(self, rules: list[Rule], random_state: RandomState, size: int = 1) -> list[Rule]:
        return random_state.choice(rules, size=size)
