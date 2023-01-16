import numpy as np
from suprb.rule import Rule
from suprb.base import BaseComponent
from scipy.spatial.distance import hamming
from .novelty_search_type import NoveltySearchType
from .archive import Archive, ArchiveNovel


class NoveltyCalculation(BaseComponent):
    def __init__(self, novelty_search_type: NoveltySearchType = NoveltySearchType(),
                 archive: Archive = ArchiveNovel(), k_neighbor: int = 15):
        self.novelty_search_type = novelty_search_type
        self.archive = archive
        self.k_neighbor = k_neighbor

    def __call__(self, rules: list[Rule]) -> list[Rule]:
        return self._novelty_score(rules=rules)

    def _novelty_score(self, rules: list[Rule]) -> list[Rule]:
        """ The basic novely calculation based on the hamming distance. 
            Takes every rule and calculates the hamming distance to each rule contained in the archive
            and averages the distances of the k-nearest neighbors to get the novelty score.
        """
        novelty_search_rules = []
        archive = self.archive.archive[:]
        filtered_rules = self.novelty_search_type.filter_rules(rules)

        for i, rule in enumerate(filtered_rules):
            if not hasattr(rule, 'idx_') or rule.idx_ > len(archive):
                rule.distances_ = []
                for archive_rule in archive:
                    hamming_distance = hamming(rule.match_set_, archive_rule.match_set_)
                    archive_rule.distances_.append(hamming_distance)
                    rule.distances_.append(hamming_distance)

                rule.distances_.append(0)
                rule.idx_ = len(archive)

            archive.append(rule)

        for i, rule in enumerate(filtered_rules):
            local_competition = self.novelty_search_type.local_competition(rule, archive)
            rule.novelty_score_ = local_competition + self._novelty_score_calculation(rule)
            novelty_search_rules.append(rule)

        return novelty_search_rules

    def _novelty_score_calculation(self, rule: Rule) -> float:
        num_neighbors = min(self.k_neighbor, len(rule.distances_) - 1)
        k_closest_neighbors = np.partition(rule.distances_, num_neighbors)[:num_neighbors]
        return sum(k_closest_neighbors) / num_neighbors


class ProgressiveMinimalCriteria(NoveltyCalculation):
    """ Uses the basic novelty score calculation, but only with rules that are above the median fitness.
        The returned list of rules will be shorter than the original input 
    """

    def _novelty_score(self, rules: list[Rule]) -> list[Rule]:
        median_fitness = np.median([rule.fitness_ for rule in rules])
        filtered_rules = [rule for rule in rules if rule.fitness_ >= median_fitness]

        return super()._novelty_score(rules=filtered_rules)


class NoveltyFitnessPareto(NoveltyCalculation):
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
    """
    Uses the basic novelty score calculation with weighted novelty and fitness.
    novelty_bias: float
        Percentage value [0, 1]
    """

    def __init__(self, novelty_bias: float = 0.5, novelty_search_type:
                 NoveltySearchType = NoveltySearchType(),
                 archive: Archive = ArchiveNovel()):
        self.novelty_bias = novelty_bias
        self.fitness_bias = 1 - novelty_bias

        super().__init__(novelty_search_type=novelty_search_type, archive=archive)

    def _novelty_score_calculation(self, rule: Rule) -> float:
        basic_novelty_score = super()._novelty_score_calculation(rule)
        scaled_fitness = rule.fitness_ / 100
        novelty_score = (self.novelty_bias * basic_novelty_score) + (self.fitness_bias * scaled_fitness)

        return novelty_score
