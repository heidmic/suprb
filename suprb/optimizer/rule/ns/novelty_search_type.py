from abc import ABCMeta, abstractmethod

import numpy as np
from suprb.rule import Rule
from suprb.base import BaseComponent


class NoveltySearchType(BaseComponent, metaclass=ABCMeta):
    """Abstract class of Novelty Search Type"""

    def filter_rules(self, rules: list[Rule]):
        pass

    def local_competition(self, rule: Rule, ns_rules: list[Rule]) -> float:
        pass


class BasicNoveltySearchType(NoveltySearchType):
    """Basic Novelty Search without any modifications"""

    def filter_rules(self, rules: list[Rule]):
        return rules

    def local_competition(self, rule: Rule, ns_rules: list[Rule]) -> float:
        return 0


class MinimalCriteria(NoveltySearchType):
    """Minimal Criteria Novelty Search, where a rule needs to be matched by a minimum amount of samples to be considered as matched"""

    def __init__(self, min_samples_matched: int):
        self.min_samples_matched = min_samples_matched

    def filter_rules(self, rules: list[Rule]) -> list[Rule]:
        # calculate the 25th percentile value and if it's lower than MNCS_threshold_matched use it instead to filter
        # a maximum of 25% of the population (to prevent empty populations)
        maximum_threshold = min(np.percentile([np.count_nonzero(rule.match_) for rule in rules],
                                              self.min_samples_matched), 25)

        return [rule for rule in rules if np.count_nonzero(rule.match_) >= maximum_threshold]


class LocalCompetition(NoveltySearchType):
    """Local Competition Novelty Search, where only rules are considered that are in the vicinity of another rule"""

    def __init__(self, max_neighborhood_range: int):
        self.max_neighborhood_range = max_neighborhood_range

    def local_competition(self, rule: Rule, ns_rules: list[Rule]) -> float:
        count_worse = 0
        for ns_rule, _ in ns_rules[:self.max_neighborhood_range]:
            if ns_rule.fitness_ < rule.fitness_:
                count_worse += 1
        local_score = count_worse / len(ns_rules[:self.max_neighborhood_range])

        return local_score
