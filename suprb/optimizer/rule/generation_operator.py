from abc import ABCMeta, abstractmethod
from suprb.rule.matching import MatchingFunction, OrderedBound
from suprb.rule import Rule
from suprb.base import BaseComponent
from suprb.utils import RandomState


class GenerationOperator(BaseComponent, metaclass=ABCMeta):
    def __init__(self, matching_type: MatchingFunction = None):
        self._matching_type = matching_type

    @property
    def matching_type(self):
        return self._matching_type

    @matching_type.setter
    def matching_type(self, matching_type):
        self._matching_type = matching_type
        if isinstance(self.matching_type, OrderedBound):
            self.execute = self.ordered_bound

    @abstractmethod
    def ordered_bound(self, rule: Rule, random_state: RandomState):
        pass
