from abc import ABCMeta, abstractmethod

from suprb.rule import Rule
from suprb.base import BaseComponent
from suprb.utils import RandomState


class Archive(BaseComponent, metaclass=ABCMeta):
    def __init__(self):
        self.archive = []

    def set_archive(self, pool: list[Rule]):
        self.archive = pool

    def set_random_state(self, random_state: RandomState):
        self.random_state_ = random_state

    def get_archive(self):
        return self.archive

    def extend_archive(self, rules: list[Rule], n: int):
        self._add_rules_to_archive(rules, n)

    def _add_rules_to_archive(self, rules: list[Rule], n: int):
        pass


class ArchiveNovel(Archive):

    def _add_rules_to_archive(self, rules: list[Rule], n: int):
        sorted_rules = sorted(rules, key=lambda x: (x.novelty_score, x.experience_), reverse=True)
        self.archive.extend([x for x in sorted_rules][:n])


class ArchiveRandom(Archive):

    def _add_rules_to_archive(self, rules: list[Rule], n: int):
        self.random_state_.shuffle(rules)
        self.archive.extend([x for x in rules][:n])


class ArchiveNone(Archive):

    def set_archive(self, pool: list[Rule]):
        self.archive = []

    def extend_archive(self, rules: list[Rule], n: int):
        self.archive = []