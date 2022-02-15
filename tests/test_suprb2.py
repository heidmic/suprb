import unittest

from sklearn.utils.estimator_checks import check_estimator

import suprb2
import suprb2.logging.stdout


class TestSupRB2(unittest.TestCase):

    def test_check_estimator(self):
        """Tests that `check_estimator()` from sklearn passes,
         i.e., that the scikit-learn interface guidelines are met."""

        # Low n_iter for speed. Still takes forever though.
        estimator = suprb2.SupRB2(
            n_iter=1,
            solution_optimizer=suprb2.optimizer.solution.ga.GeneticAlgorithm(n_iter=16, population_size=16),
            logger=suprb2.logging.stdout.StdoutLogger(),
            verbose=10
        )

        check_estimator(estimator)
