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
    return np.sum([hamming_distance(a.genome, b.genome) for a, b in combinations]) / population[0].genome.shape[0]


def hamming_distance(a: np.ndarray, b: np.ndarray):
    return np.count_nonzero(a ^ b)


def matched_training_samples(pool: list[Rule]):
    if not pool:
        return 0

    matched = np.stack([rule.match_set_ for rule in pool]).any(axis=0).nonzero()[0].shape[0]
    total = pool[0].match_set_.shape[0]
    return matched / total


def hypervolume(pareto_front: np.ndarray, reference_point: np.ndarray = None):
    """Assumes that the pareto front is sorted by complexity in ascending order."""
    last_x = reference_point[0]
    volume = 0
    for fitness in pareto_front:
        volume += (last_x - fitness[0]) * np.prod(reference_point[1:] - fitness[1:])
        last_x = fitness[0]
    return volume


def spread(pareto_front: np.ndarray):
    if len(pareto_front) <= 1:
        return 0
    """Assumes that the pareto front is sorted by complexity in ascending order."""
    distances = np.linalg.norm(pareto_front[:-1] - pareto_front[1:], axis=1)
    avg_distance = np.mean(distances)
    np.sum(np.abs(distances - avg_distance) / distances.shape[0] - 1)
    return np.max(distances)
