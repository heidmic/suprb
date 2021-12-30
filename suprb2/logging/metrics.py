import itertools
import numpy as np

from suprb2 import Rule, Individual


def genome_diversity(population: list[Individual]):
    combinations = list(itertools.combinations(population, 2))
    return sum([hamming_distance(a.genome, b.genome) for a, b in combinations])


def hamming_distance(a: np.ndarray, b: np.ndarray):
    return np.count_nonzero(a ^ b)


def matched_training_samples(pool: list[Rule]):
    if not pool:
        return 0

    matched = np.stack([rule.match_ for rule in pool]).any(axis=0).nonzero()[0].shape[0]
    total = pool[0].match_.shape[0]
    return matched / total
