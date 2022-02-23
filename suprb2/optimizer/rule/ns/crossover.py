from abc import ABCMeta, abstractmethod

import numpy as np
from suprb2.rule import Rule
from suprb2.base import BaseComponent
from suprb2.utils import RandomState


class RuleCrossover(BaseComponent, metaclass=ABCMeta):

    def __init__(self, crossover_rate: float = 0.2):
        self.crossover_rate = crossover_rate

    def __call__(self, A: Rule, B: Rule, random_state: RandomState) -> list[Rule]:
        if random_state.random() < self.crossover_rate:
            return self._crossover(A=A, B=B, random_state=random_state)
        else:
            # return unmodified parents
            return [A, B]

    @abstractmethod
    def _crossover(self, A: Rule, B: Rule, random_state: RandomState) -> list[Rule]:
        pass


class UniformCrossover(RuleCrossover):
    """Decide for every bound tuple with uniform probability if the bound tuple in rule A or B is used."""

    def _crossover(self, A: Rule, B: Rule, random_state: RandomState) -> list[Rule]:
        # For each dimension both parents have equal chances of supplying the respective pair of bounds. This equates
        # to a threshold of .5 for the random generator values.
        indices = random_state.random(size=len(A.bounds)) <= 0.5
        a = A.clone()
        b = B.clone()
        for i in range(len(indices)):
            if indices[i]:
                a.bounds[i] = B.bounds[i]
                b.bounds[i] = A.bounds[i]

        return [a, b]
