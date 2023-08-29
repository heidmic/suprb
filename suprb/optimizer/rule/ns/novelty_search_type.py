import numpy as np
from suprb.rule import Rule
from suprb.base import BaseComponent


class NoveltySearchType(BaseComponent):
    """Basic Novelty Search without any modifications"""

    def filter_rules(self, rules: list[Rule]):
        return rules

    def local_competition(self, rule: Rule, rules: list[Rule]) -> float:
        return 0


class MinimalCriteria(NoveltySearchType):
    """Minimal Criteria Novelty Search, where a rule needs to be matched by a minimum amount of examples to be considered as matched"""

    def __init__(self, min_examples_matched: int = 15):
        self.min_examples_matched = min_examples_matched

    def filter_rules(self, rules: list[Rule]) -> list[Rule]:
        # calculate the 25th percentile value and if it's lower than MNCS_threshold_matched use it instead to filter
        # a maximum of 25% of the population (to prevent empty populations)
        maximum_threshold = min(np.percentile([np.count_nonzero(rule.match_set_) for rule in rules],
                                              self.min_examples_matched), 25)

        return [rule for rule in rules if np.count_nonzero(rule.match_set_) >= maximum_threshold]


class LocalCompetition(NoveltySearchType):
    """Local Competition Novelty Search, where only rules are considered that are in the vicinity of another rule"""

    def __init__(self, max_neighborhood_range: int = 15):
        self.max_neighborhood_range = max_neighborhood_range

    def local_competition(self, rule: Rule, rules: list[Rule]) -> float:
        count_worse = 0
        for archive_rule in rules[:self.max_neighborhood_range]:
            if rule.fitness_ < archive_rule.fitness_:
                count_worse += 1
        local_score = count_worse / self.max_neighborhood_range

        return local_score
