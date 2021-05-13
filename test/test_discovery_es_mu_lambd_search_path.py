from suprb2.discovery import ES_MuLambdSearchPath
from suprb2.classifier import Classifier
from test.tests_support import TestsSupport

import unittest

class TestDiscoveryES_MuLambdSearchPath(unittest.TestCase):
    """
    This module test all methods from ES_MuLambdSearchPath
    """


    # ------------- step() --------------


    def test_step_mu_equal_lmbd(self):
        """
        Tests the method ES_MuLambdSearchPath.step().

        If we select lmbd <= mu, then the method
        select_best_classifiers will raise ValueError
        """
        mu, lmbd = (15, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, steps_per_step=4)
        X, y = TestsSupport.generate_input(mu)

        optimizer = ES_MuLambdSearchPath(pool=[])
        self.assertRaises(ValueError, optimizer.step, X, y)


    def test_step_mu_bigger_than_population(self):
        """
        Tests the method ES_MuLambdSearchPath.step().

        If mu is bigger than the population's size, then raise
        Exception (ValueError).
        """
        mu, lmbd = (15, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, steps_per_step=4)
        X, y = TestsSupport.generate_input(mu - 5)

        optimizer = ES_MuLambdSearchPath(pool=[])
        self.assertRaises(ValueError, optimizer.step, X, y)


    def test_step_lambd_zero(self):
        """
        Tests the method ES_MuLambdSearchPath.step().

        When lmbd is zero, then method will raise IndexError.
        """
        mu, lmbd = (15, 0)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, steps_per_step=4)
        X, y = TestsSupport.generate_input(mu)

        optimizer = ES_MuLambdSearchPath(pool=[])
        self.assertRaises(IndexError, optimizer.step, X, y)


    def test_step_mu_zero(self):
        """
        Tests the method ES_MuLambdSearchPath.step().

        When mu is zero, then method will raise IndexError.
        """
        mu, lmbd = (0, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, steps_per_step=4)
        X, y = TestsSupport.generate_input(10)

        optimizer = ES_MuLambdSearchPath(pool=[])
        with self.assertWarns(RuntimeWarning):
            self.assertRaises(IndexError, optimizer.step, X, y)


    def test_step_no_input(self):
        """
        Tests the method ES_MuLambdSearchPath.step().

        If our X is empty (no data is given),
        we can still create (unfit) classifiers
        at the rate of mu * steps_per_step.
        """
        mu, lmbd, steps_per_step = (10, 15, 4)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, steps_per_step=steps_per_step)
        X, y = TestsSupport.generate_input(0)

        optimizer = ES_MuLambdSearchPath(pool=[])

        with self.assertWarns(RuntimeWarning):
            optimizer.step(X, y)
        self.assertEqual(len(optimizer.pool), mu * steps_per_step)



    # ------------- select_best_classifiers() --------------


    def test_select_best_classifiers(self):
        """
        This method tests the method ES_MuLambdSearchPath.select_best_classifiers()

        Checks that the resulting array has 'mu' elements and that they are, indeed,
        the best 'mu'.
        """
        n, mu = (10, 5)
        optimizer = ES_MuLambdSearchPath(pool=[])
        X, y = TestsSupport.generate_input(n)
        pool = list()
        for i in range(n):
            cl = Classifier.random_cl(point=X[i], xdim=X.shape[1])
            cl.error = i
            pool.append([cl, optimizer.create_sigmas(X.shape[1])])

        best_tuples = optimizer.select_best_classifiers(pool, mu)
        self.assertEqual(len(best_tuples), mu)
        self.assertEqual(len(best_tuples[0]), 2)


    def test_select_best_classifiers_mu_equal_tuple_list(self):
        """
        This method tests the method ES_MuLambdSearchPath.select_best_classifiers()

        Checks that method raises ValueError() when mu == len(tuple_list)
        """
        mu = 5
        optimizer = ES_MuLambdSearchPath(pool=[])
        X, y = TestsSupport.generate_input(mu)
        pool = list()
        for i in range(mu):
            cl = Classifier.random_cl(point=X[i], xdim=X.shape[1])
            cl.error = i
            pool.append([cl, optimizer.create_sigmas(X.shape[1])])

        self.assertRaises(ValueError, optimizer.select_best_classifiers, pool, mu)
        self.assertRaises(ValueError, optimizer.select_best_classifiers, pool, mu + 1)


if __name__ == '__main__':
    unittest.main()
