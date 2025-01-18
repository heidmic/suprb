import itertools
import numpy as np

from suprb import Rule, Solution


def genome_diversity(population: list[Solution]):
    """
    Calculates the relative pairwise hamming distance of the genomes of all solutions in the population.

    Note that this does not speak to the diversity of individual solutions, i.e. two different genomes can construct
    identical solutions if the same rule is part of the pool multiple times
    """

    combinations = list(itertools.combinations(population, 2))
    return (
        np.sum([hamming_distance(a.genome, b.genome) for a, b in combinations])
        / population[0].genome.shape[0]
    )


def hamming_distance(a: np.ndarray, b: np.ndarray):
    return np.count_nonzero(a ^ b)


def matched_training_samples(pool: list[Rule]):
    if not pool:
        return 0

    matched = (
        np.stack([rule.match_set_ for rule in pool]).any(axis=0).nonzero()[0].shape[0]
    )
    total = pool[0].match_set_.shape[0]
    return matched / total
