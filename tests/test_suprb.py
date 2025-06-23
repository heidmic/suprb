import unittest
import numpy as np

from sklearn.utils.estimator_checks import check_estimator, _regression_dataset

import suprb
import suprb.logging.stdout

from suprb import SupRB
from suprb.rule import Rule
from suprb.rule.fitness import VolumeWu
from suprb.rule.matching import OrderedBound
from suprb.optimizer.rule.es import ES1xLambda


class TestSupRB(unittest.TestCase):

    def create_rule(self, fitness, experience, error):
        rule = Rule(
            match=OrderedBound(np.array([[-1, 1]])),
            input_space=[-1.0, 1.0],
            model=SupRB(),
            fitness=VolumeWu,
        )

        rule.fitness_ = fitness
        rule.experience_ = experience
        rule.error_ = error

        return rule

    def test_check_estimator(self):
        """Tests that `check_estimator()` from sklearn passes,
        i.e., that the scikit-learn interface guidelines are met."""

        # Low n_iter for speed. Still takes forever though.
        estimator = suprb.SupRB(
            n_iter=4,
            rule_discovery=ES1xLambda(n_iter=4, lmbda=1, delay=2),
            solution_composition=suprb.optimizer.solution.ga.GeneticAlgorithm(n_iter=2, population_size=2),
            logger=suprb.logging.stdout.StdoutLogger(),
            verbose=10,
        )

        X, y = _regression_dataset()
        estimator.fit(X, y)

        check_estimator(estimator)

    def test_early_stopping(self):
        estimator = suprb.SupRB(
            n_iter=1,
            rule_discovery=ES1xLambda(n_iter=4, lmbda=1, delay=2),
            solution_composition=suprb.optimizer.solution.ga.GeneticAlgorithm(n_iter=2, population_size=2),
            logger=suprb.logging.stdout.StdoutLogger(),
            verbose=10,
            early_stopping_patience=3,
            early_stopping_delta=10,
        )

        # Setup necessary parameter for early stopping test
        estimator._validate_solution_composition()
        estimator.early_stopping_counter_ = 0

        # early_stopping_counter_ is increased to 1 [previous_fitness_(4) > current_fitness(3)]
        estimator.previous_fitness_ = 4
        estimator.solution_composition_.population_ = [self.create_rule(3, 2, 2)]
        self.assertFalse(estimator.check_early_stopping())

        # early_stopping_counter_ is reset to 0 [previous_fitness_(4) < current_fitness(20)]
        estimator.previous_fitness_ = 4
        estimator.solution_composition_.population_ = [self.create_rule(20, 2, 2)]
        self.assertFalse(estimator.check_early_stopping())

        # early_stopping_counter_ is increased to 1 [previous_fitness_(20) > current_fitness(5)]
        estimator.previous_fitness_ = 20
        estimator.solution_composition_.population_ = [self.create_rule(5, 2, 2)]
        self.assertFalse(estimator.check_early_stopping())

        # early_stopping_counter_ is increased to 2 [previous_fitness_(20) > current_fitness(10)]
        estimator.previous_fitness_ = 20
        estimator.solution_composition_.population_ = [self.create_rule(10, 2, 2)]
        self.assertFalse(estimator.check_early_stopping())

        # early_stopping_counter_ is increased to 3 [current_fitness(22) - previous_fitness_(20) < early_stopping_delta(10)]
        estimator.previous_fitness_ = 20
        estimator.solution_composition_.population_ = [self.create_rule(22, 2, 2)]
        self.assertTrue(estimator.check_early_stopping())
