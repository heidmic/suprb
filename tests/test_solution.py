import unittest

from sklearn.utils.estimator_checks import check_estimator, _regression_dataset

import suprb
import suprb.logging.stdout
from suprb.optimizer.solution.ga import GeneticAlgorithm
from suprb.optimizer.solution.saga import (
    SelfAdaptingGeneticAlgorithm1,
    SelfAdaptingGeneticAlgorithm2,
    SelfAdaptingGeneticAlgorithm3,
    SasGeneticAlgorithm,
)
from suprb.optimizer.rule.es import ES1xLambda


class TestSolution(unittest.TestCase):

    def test_check_ga(self):
        estimator = suprb.SupRB(
            n_iter=4,
            rule_discovery=ES1xLambda(n_iter=4, lmbda=1, delay=2),
            solution_composition=GeneticAlgorithm(n_iter=2, population_size=2),
            logger=suprb.logging.stdout.StdoutLogger(),
            verbose=10,
        )

        X, y = _regression_dataset()
        estimator.fit(X, y)

        check_estimator(estimator)

    def test_check_saga1(self):
        estimator = suprb.SupRB(
            n_iter=4,
            rule_discovery=ES1xLambda(n_iter=4, lmbda=1, delay=2),
            solution_composition=SelfAdaptingGeneticAlgorithm1(n_iter=2, population_size=2),
            logger=suprb.logging.stdout.StdoutLogger(),
            verbose=10,
        )
        X, y = _regression_dataset()
        estimator.fit(X, y)

        check_estimator(estimator)

    def test_check_saga2(self):
        estimator = suprb.SupRB(
            n_iter=4,
            rule_discovery=ES1xLambda(n_iter=4, lmbda=1, delay=2),
            solution_composition=SelfAdaptingGeneticAlgorithm2(n_iter=2, population_size=2),
            logger=suprb.logging.stdout.StdoutLogger(),
            verbose=10,
        )

        X, y = _regression_dataset()
        estimator.fit(X, y)

        check_estimator(estimator)

    def test_check_saga3(self):
        estimator = suprb.SupRB(
            n_iter=4,
            rule_discovery=ES1xLambda(n_iter=4, lmbda=1, delay=2),
            solution_composition=SelfAdaptingGeneticAlgorithm3(n_iter=2, population_size=2),
            logger=suprb.logging.stdout.StdoutLogger(),
            verbose=10,
        )

        X, y = _regression_dataset()
        estimator.fit(X, y)

        check_estimator(estimator)

    def test_check_sas(self):
        estimator = suprb.SupRB(
            n_iter=4,
            rule_discovery=ES1xLambda(n_iter=4, lmbda=1, delay=2),
            solution_composition=SasGeneticAlgorithm(n_iter=2, initial_population_size=2),
            logger=suprb.logging.stdout.StdoutLogger(),
            verbose=10,
        )

        X, y = _regression_dataset()
        estimator.fit(X, y)

        check_estimator(estimator)
