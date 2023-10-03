import numpy as np

from suprb.optimizer.solution.sas.solution_extension import SasSolution
from suprb.rule import Rule
from suprb.solution.initialization import SolutionInit, MixingModel, SolutionFitness
from ..utils import RandomState


def padding_size(solution: SasSolution) -> int:
    """Calculates the number of bits to add to the genome after the pool was expanded."""

    return len(solution.pool) - solution.genome.shape[0]


def random(n: int, p: float, random_state: RandomState):
    """Returns a random bit string of size `n`, with ones having probability `p`."""

    return (random_state.random(size=n) <= p).astype('bool')


class SasRandomInit(SolutionInit):
    """Init and extend genomes with random values, with `p` denoting the probability of ones."""

    def __init__(self, mixing: MixingModel = None, fitness: SolutionFitness = None, p: float = 0.5):
        super().__init__(mixing=mixing, fitness=fitness)
        self.p = p

    def __call__(self, pool: list[Rule], random_state: RandomState) -> SasSolution:
        return SasSolution(genome=random(len(pool), self.p, random_state), pool=pool, mixing=self.mixing,
                        fitness=self.fitness)

    def pad(self, solution: SasSolution, random_state: RandomState) -> SasSolution:
        solution.genome = np.concatenate((solution.genome, random(padding_size(solution), self.p, random_state)),
                                         axis=0)
        return solution
