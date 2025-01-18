from abc import ABCMeta, abstractmethod
import numpy as np
from suprb.rule.matching import (
    MatchingFunction,
    OrderedBound,
    UnorderedBound,
    CenterSpread,
    MinPercentage,
)
from suprb.rule import Rule
from suprb.base import BaseComponent
from suprb.utils import RandomState


class GenerationOperator(BaseComponent, metaclass=ABCMeta):
    def __init__(self, matching_type: MatchingFunction = None):
        self.matching_type = matching_type

    @property
    def matching_type(self):
        return self._matching_type

    @matching_type.setter
    def matching_type(self, matching_type):
        self._matching_type = matching_type
        if isinstance(self.matching_type, OrderedBound):
            self.execute = self.ordered_bound
        elif isinstance(self.matching_type, UnorderedBound):
            self.execute = self.unordered_bound
        elif isinstance(self.matching_type, CenterSpread):
            self.execute = self.center_spread
            assert isinstance(self.sigma, np.ndarray) and self.sigma.shape[0] == 2
        elif isinstance(self.matching_type, MinPercentage):
            self.execute = self.min_percentage
            assert isinstance(self.sigma, np.ndarray) and self.sigma.shape[0] == 2

    @abstractmethod
    def ordered_bound(self, rule: Rule, random_state: RandomState):
        pass

    @abstractmethod
    def unordered_bound(self, rule: Rule, random_state: RandomState):
        pass

    @abstractmethod
    def center_spread(self, rule: Rule, random_state: RandomState):
        pass

    @abstractmethod
    def min_percentage(self, rule: Rule, random_state: RandomState):
        pass
