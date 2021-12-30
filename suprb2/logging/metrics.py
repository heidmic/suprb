import itertools
import numpy as np

from suprb2 import Rule, Individual


def genome_diversity(population: list[Individual]):
    combinations = list(itertools.combinations(population, 2))
    return sum([hamming_distance(com[0].genome, com[1].genome) for com in combinations])


def hamming_distance(a: np.ndarray, b: np.ndarray):
    r = (1 << np.arange(8))[:, None]
    return np.count_nonzero((np.bitwise_xor(a, b) & r) != 0)


def matched_training_samples(pool: list[Rule]):
    if not pool:
        return 0

    matched = np.stack([rule.match_ for rule in pool]).any(axis=0).nonzero()[0].shape[0]
    total = pool[0].match_.shape[0]
    return matched / total
