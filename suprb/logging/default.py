from collections import defaultdict

import numpy as np

from . import BaseLogger
from .metrics import matched_training_samples, genome_diversity
from .. import SupRB


class DefaultLogger(BaseLogger):
    """Stores relevant parameters and metrics in memory."""

    params_: dict
    metrics_: dict
    genomes_: dict
    pool_: dict

    def log_param(self, key, value):
        self.params_[key] = value

    def log_metric(self, key, value, step):
        self.metrics_[key][step] = value

    def log_params(self, **kwargs):
        for key, value in kwargs.items():
            self.log_param(key=key, value=value)

    def log_init(self, X: np.ndarray, y: np.ndarray, estimator: SupRB):
        self.params_ = {}
        self.metrics_ = defaultdict(dict)
        self.genomes_ = {}
        self.log_params(**estimator.get_params())

    def log_iteration(self, X: np.ndarray, y: np.ndarray, estimator: SupRB, iteration: int):
        def log_metric(key, value):
            self.log_metric(key=key, value=value, step=estimator.step_)

        def log_metric_min_max_mean(metric_name: str, attribute_name: str, lst: list):
            comprehension = [getattr(e, attribute_name) for e in lst]
            log_metric(metric_name + '_min', min(comprehension))
            log_metric(metric_name + '_mean', sum(comprehension) / len(comprehension))
            log_metric(metric_name + '_max', max(comprehension))

        # Log pool
        pool = estimator.pool_
        log_metric("pool_size", len(pool))
        log_metric("pool_matched", matched_training_samples(pool))
        log_metric_min_max_mean("pool_fitness", 'fitness_', pool)

        # Log population
        # Note that this technically is `PopulationBasedSolutionComposition` specific.
        population = estimator.solution_composition_.population_
        log_metric("population_diversity", genome_diversity(population))
        log_metric_min_max_mean("population_fitness", 'fitness_', population)
        log_metric_min_max_mean("population_error", 'error_', population)
        log_metric_min_max_mean("population_complexity", 'complexity_', population)

        # Log elitist
        elitist = estimator.solution_composition_.elitist()
        log_metric("elitist_fitness", elitist.fitness_)
        log_metric("elitist_error", elitist.error_)
        log_metric("elitist_complexity", elitist.complexity_)
        log_metric("elitist_matched", matched_training_samples(elitist.subpopulation))

        # Log performance
        log_metric("training_score", elitist.score(X, y))

    def log_final(self, X: np.ndarray, y: np.ndarray, estimator: SupRB):
        def log_metric(key, value):
            self.log_metric(key=key, value=value, step=estimator.step_)

        def log_metric_min_max_mean(metric_name: str, lst: list):
            log_metric(metric_name + '_min', min(lst))
            log_metric(metric_name + '_mean', sum(lst) / len(lst))
            log_metric(metric_name + '_max', max(lst))

        def parse_genome(genome_lst):
            str_genomes = []
            for x in genome_lst:
                str_genomes.append(''.join((str(int(e))) for e in x))
            return str_genomes

        def get_convergence(lst: list, threshold: int):
            copied_lst = lst.copy()
            final_genome = copied_lst.pop()
            length_final = final_genome.shape[0]
            for i, gen in zip(range(len(copied_lst)), reversed(copied_lst)):
                thresh = 0
                length_current = gen.shape[0]
                for x in range(length_final):
                    if x > length_current - 1:
                        if final_genome[x] != 0:
                            thresh += 1
                    elif final_genome[x] != gen[x]:
                        thresh += 1
                    if thresh > threshold:
                        return i
            return len(lst) - 1

        def get_fit_convergence(lst: list, threshold: float):
            copied_lst = lst.copy()
            final_elitist = copied_lst.pop()
            thresh = 1 + threshold
            for i, elitist in zip(range(len(copied_lst)), reversed(copied_lst)):
                if final_elitist.fitness_ > (elitist.fitness_ * thresh):
                    return i
            return len(copied_lst)

        # Log delay
        log_metric_min_max_mean("delay", estimator.final_iterations)

        # Log elitist convergence
        genomes = [o.genome for o in estimator.global_elitists]
        str_genomes = parse_genome(genomes)
        genome_dict = {i: str_genomes[i] for i in range(0, len(str_genomes))}
        self.genomes_ = genome_dict
        log_metric("elitist_convergence_thresh_0", get_convergence(genomes, 0))
        log_metric("elitist_convergence_thresh_1", get_convergence(genomes, 1))
        log_metric("elitist_convergence_thresh_2", get_convergence(genomes, 2))

        # Log fitness convergence
        log_metric("fit_convergence_0", get_fit_convergence(estimator.global_elitists, 0.01))
        log_metric("fit_convergence_1", get_fit_convergence(estimator.global_elitists, 0.001))
        log_metric("fit_convergence_2", get_fit_convergence(estimator.global_elitists, 0.0001))
        log_metric("fit_convergence_3", get_fit_convergence(estimator.global_elitists, 0.00001))
