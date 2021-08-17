from suprb2.discovery import ES_MuLambdSearchPath
from test.tests_support import TestsSupport

import unittest


class TestDiscoveryES_MuLambdSearchPath(unittest.TestCase):
    """
    This module test all methods from ES_MuLambdSearchPath
    """

    def test_step_mu_equal_lmbd(self):
        """
        Tests the method ES_MuLambdSearchPath.step().

        lmbd == mu should not be a problem, since we
        create points using one start point.
        """
        mu, lmbd = (15, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, steps_per_step=4)
        X, y = TestsSupport.generate_input(mu)

        optimizer = ES_MuLambdSearchPath(pool=[])
        optimizer.step(X, y)
        self.assertEqual(len(optimizer.pool), lmbd * 4)

    def test_step_mu_bigger_than_lmbd(self):
        """
        Tests the method ES_MuLambdSearchPath.step().

        mu > lmbd is a problem, because we call
        select_best_classifier with mu > len(cls_tuples).
        This behaviour shouldn't be allowed as well,
        in order to avoid mistakes in the future.
        """
        mu, lmbd = (15, 10)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, steps_per_step=4)
        X, y = TestsSupport.generate_input(mu)

        optimizer = ES_MuLambdSearchPath(pool=[])
        with self.assertRaises(ValueError) as cm:
            optimizer.step(X, y)
        self.assertEqual('kth(=14) out of bounds (10)', str(cm.exception))

    def test_step_mu_bigger_than_population(self):
        """
        Tests the method ES_MuLambdSearchPath.step().

        mu is bigger than the population's size should
        not be a problem, since we create points using one
        start point.
        """
        mu, lmbd = (15, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, steps_per_step=4)
        X, y = TestsSupport.generate_input(mu - 5)

        optimizer = ES_MuLambdSearchPath(pool=[])
        optimizer.step(X, y)
        self.assertEqual(len(optimizer.pool), lmbd * 4)

    def test_step_lambd_zero(self):
        """
        Tests the method ES_MuLambdSearchPath.step().

        When lmbd is zero, then method will raise IndexError.
        """
        mu, lmbd = (15, 0)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, steps_per_step=4)
        X, y = TestsSupport.generate_input(mu)

        optimizer = ES_MuLambdSearchPath(pool=[])
        with self.assertRaises(IndexError) as cm:
            optimizer.step(X, y)
        self.assertEqual('too many indices for array: array is 1-dimensional, but 2 were indexed', str(cm.exception))

    def test_step_mu_zero(self):
        """
        Tests the method ES_MuLambdSearchPath.step().

        When mu is zero, then method will raise IndexError.
        """
        mu, lmbd = (0, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, steps_per_step=4)
        X, y = TestsSupport.generate_input(10)

        optimizer = ES_MuLambdSearchPath(pool=[])
        with self.assertRaises(IndexError) as cm:
            optimizer.step(X, y)
        self.assertEqual('too many indices for array: array is 1-dimensional, but 2 were indexed', str(cm.exception))

    def test_step_no_input(self):
        """
        Tests the method ES_MuLambdSearchPath.step().

        If our X is empty (no data is given),
        raise ValueError.
        """
        mu, lmbd, steps_per_step = (10, 15, 4)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, steps_per_step=steps_per_step)
        X, y = TestsSupport.generate_input(0)

        optimizer = ES_MuLambdSearchPath(pool=[])
        with self.assertRaises(ValueError) as cm:
            optimizer.step(X, y)
        self.assertEqual('a cannot be empty unless no samples aretaken', str(cm.exception))
        # optimizer.step(X, y)
        # self.assertEqual(len(optimizer.pool), mu * steps_per_step)


if __name__ == '__main__':
    unittest.main()
