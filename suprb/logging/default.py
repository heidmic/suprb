from collections import defaultdict

import json
import numpy as np

from . import BaseLogger
from .metrics import matched_training_samples, genome_diversity
from .. import json
from suprb.base import BaseSupervised


class DefaultLogger(BaseLogger):
    """Stores relevant parameters and metrics in memory."""

    params_: dict
    metrics_: dict

    def log_param(self, key, value):
        self.params_[key] = value

    def log_metric(self, key, value, step):
        self.metrics_[key][step] = value

    def log_params(self, **kwargs):
        for key, value in kwargs.items():
            self.log_param(key=key, value=value)

    def log_init(self, X: np.ndarray, y: np.ndarray, estimator: BaseSupervised):
        self.params_ = {}
        self.metrics_ = defaultdict(dict)

        self.log_params(**estimator.get_params())

    def log_iteration(self, X: np.ndarray, y: np.ndarray, estimator: BaseSupervised, iteration: int):
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
        if pool:
            log_metric_min_max_mean("pool_fitness", 'fitness_', pool)

        # Log population
        # Note that this technically is `PopulationBasedSolutionComposition` specific.
        population = estimator.solution_composition_.population_
        log_metric("population_size", len(population))
        log_metric("population_diversity", genome_diversity(population))
        log_metric_min_max_mean("population_fitness", 'fitness_', population)
        log_metric_min_max_mean("population_error", 'error_', population)
        log_metric_min_max_mean("population_complexity", 'complexity_', population)

        # Log elitist
        elitist = estimator.solution_composition_.elitist()
        log_metric("elitist_fitness", elitist.fitness_)
        log_metric("elitist_error", elitist.error_)
        if elitist.isClassifier:
            log_metric("elitist_accuracy", elitist.score(X, y))
            log_metric("elitist_f1", elitist.f1_score(X, y))
        log_metric("elitist_complexity", elitist.complexity_)
        log_metric("elitist_matched", matched_training_samples(elitist.subpopulation))
        log_metric("elitist_rules", elitist.pool)

        # Log performance
        log_metric("training_score", elitist.score(X, y))

    def get_elitist(self, estimator: BaseSupervised):
        json_data = {}
        json._save_pool(estimator.solution_composition_.elitist().pool, json_data)
        return json_data

    def log_final(self, X: np.ndarray, y: np.ndarray, estimator: BaseSupervised):
        pass
