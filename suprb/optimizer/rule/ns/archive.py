from abc import ABCMeta, abstractmethod

from suprb.rule import Rule
from suprb.base import BaseComponent
from suprb.utils import RandomState


class Archive(BaseComponent, metaclass=ABCMeta):
    """Base Archive Class to store the rules from previous populations"""

    def __call__(self, rules: list[Rule], n: int):
        """Adds n rules to the existing archive. Rules added depend on archive type"""
        self._add_rules_to_archive(rules, n)

    @property
    def archive(self):
        return self._archive

    @archive.setter
    def archive(self, archive: list[Rule]):
        self._archive = archive

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, random_state: RandomState):
        self._random_state = random_state

    @abstractmethod
    def _add_rules_to_archive(self, rules: list[Rule], n: int):
        pass


class ArchiveNovel(Archive):
    """Archive Class that adds n rules with the best novelty scores to the archive"""

    def _add_rules_to_archive(self, rules: list[Rule], n: int):
        sorted_rules = sorted(rules, key=lambda x: (x.novelty_score_, x.experience_), reverse=True)
        self.archive.extend([x for x in sorted_rules][:n])


class ArchiveRandom(Archive):
    """Archive Class that adds n random rules to the archive"""

    def _add_rules_to_archive(self, rules: list[Rule], n: int):
        self.random_state.shuffle(rules)
        self.archive.extend([x for x in rules][:n])


class ArchiveNone(Archive):
    """
    Archive Class where no archive is saved.
    Since we don't want to use an archive, the set_archive and extend_archive functions
    """

    def _add_rules_to_archive(self, rules: list[Rule], n: int):
        pass
