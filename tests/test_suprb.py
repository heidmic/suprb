import unittest

from sklearn.utils.estimator_checks import check_estimator

import suprb
import suprb.logging.stdout


class TestSupRB(unittest.TestCase):

    def test_check_estimator(self):
        """Tests that `check_estimator()` from sklearn passes,
        i.e., that the scikit-learn interface guidelines are met."""

        # Low n_iter for speed. Still takes forever though.
        estimator = suprb.SupRB(
            n_iter=1,
            solution_composition=suprb.optimizer.solution.ga.GeneticAlgorithm(n_iter=2, population_size=2),
            logger=suprb.logging.stdout.StdoutLogger(),
            verbose=10,
        )

        check_estimator(estimator)
