from abc import ABCMeta

from suprb import Solution
from suprb.base import BaseComponent
from suprb.utils import RandomState


class AntSelection(BaseComponent, metaclass=ABCMeta):

    def __call__(
        self, population: list[Solution], random_state: RandomState
    ) -> list[Solution]:
        pass


class NBest(AntSelection):

    def __init__(self, n: int = 1):
        self.n = n

    def __call__(
        self, population: list[Solution], random_state: RandomState
    ) -> list[Solution]:
        return list(sorted(population, key=lambda i: i.fitness_, reverse=True))[
            : self.n
        ]
