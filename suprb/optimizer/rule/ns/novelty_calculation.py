from abc import ABCMeta, abstractmethod

import numpy as np
from suprb.rule import Rule
from suprb.base import BaseComponent
from scipy.spatial.distance import hamming
from .novelty_search_type import NoveltySearchType
from .archive import Archive


class NoveltyCalculation(BaseComponent, metaclass=ABCMeta):
    def __init__(self, novelty_search_type: NoveltySearchType, archive: Archive, k_neighbor: int):
        self.novelty_search_type = novelty_search_type
        self.archive = archive
        self.k_neighbor = k_neighbor

    def __call__(self, rules: list[Rule]) -> list[Rule]:
        return self._novelty_score(rules=rules)

    def _add_distances_to_archive_rules(self, rule: Rule, archive: list[Rule]):
        for i, _ in enumerate(archive):
            archive[i].distance_ = hamming(rule.match_, archive[i].match_)

    def _novelty_score(self, rules: list[Rule]) -> list[Rule]:
        """ The basic novely calculation based on the hamming distance. 
            Takes every rule and calculates the hamming distance to each rule contained in the archive
            and averages the distances of the k-nearest neighbors to get the novelty score.
        """
        novelty_search_rules = []
        archive = self.archive.get_archive()[:]
        archive.extend(rules)

        for rule in self.novelty_search_type.filter_rules(rules):
            self._add_distances_to_archive_rules(rule, archive)

            archive = sorted(archive, key=lambda archive_rule: archive_rule.distance_)
            distances = [sorted_rule.distance_ for sorted_rule in archive]

            local_competition = self.novelty_search_type.local_competition(rule, archive)
            kwargs = {"distances": distances, "fitness": rule.fitness_}
            novelty_score = local_competition + self._novelty_score_calculation(**kwargs)

            rule.novelty_score_ = novelty_score
            novelty_search_rules.append(rule)

        return novelty_search_rules

    def _novelty_score_calculation(self, **kwargs: dict()):
        return sum(kwargs["distances"][:self.k_neighbor]) / self.k_neighbor


class ProgressiveMinimalCriteria(NoveltyCalculation):
    """ Uses the basic novelty score calculation, but only with rules that are above the median fitness.
        The returned list of rules will be shorter than the original input 
    """

    def _novelty_score(self, rules: list[Rule]) -> list[Rule]:
        median_fitness = np.median([rule.fitness_ for rule in rules])
        filtered_rules = [rule for rule in rules if rule.fitness_ >= median_fitness]

        return super()._novelty_score(rules=filtered_rules)


class NovelityFitnessPareto(NoveltyCalculation):
    """Uses the basic novelty score calculation and return only the pareto front of it."""

    def _novelty_score(self, rules: list[Rule]) -> list[Rule]:
        novelty_search_rules = super()._novelty_score(rules=rules)

        return self._get_pareto_front(novelty_search_rules)

    def _get_pareto_front(self, rules: list[Rule]) -> list[Rule]:
        rules = sorted(rules,
                       key=lambda rule: (rule.novelty_score_, rule.fitness_),
                       reverse=True)

        pareto_front = [rules[0]]

        for rule in rules[1:]:
            if rule.fitness_ >= pareto_front[-1].fitness_:
                pareto_front.append(rule)

        return pareto_front


class NoveltyFitnessBiased(NoveltyCalculation):
    """Uses the basic novelty score calculation with weighted novelty and fitness."""

    def __init__(self, novelty_bias: float, novelty_search_type: NoveltySearchType, archive: Archive):
        self.novelty_bias = novelty_bias
        self.fitness_bias = 1 - novelty_bias

        super().__init__(novelty_search_type=novelty_search_type, archive=archive, k_neighbor=1)

    def _novelty_score_calculation(self, **kwargs: dict()):
        basic_novelty_score = super()._novelty_score_calculation(**kwargs)
        scaled_fitness = kwargs["fitness"] / 100
        novelty_score = (self.novelty_bias * basic_novelty_score) + (self.fitness_bias * scaled_fitness)

        return novelty_score
