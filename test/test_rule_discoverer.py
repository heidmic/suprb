from test.tests_support import TestsSupport
from suprb2.discovery import RuleDiscoverer
from suprb2.individual import Individual
from suprb2.classifier import Classifier
from suprb2.solutions import ES_1plus1

import numpy as np
import unittest

class TestDiscoveryRuleDiscoverer(unittest.TestCase):
    """
    This module test all methods from RuleDiscoverer
    """


    def setup_test(self, x_len, n, start_points, elitist):
        TestsSupport.set_rule_discovery_configs(start_points=start_points)
        X, y = TestsSupport.generate_input(x_len)
        if elitist is not None:
            solution_opt = ES_1plus1(X, y, list(), elitist)
            return RuleDiscoverer(pool=list(), solution_optimizer=solution_opt).create_start_tuples(n, X, y)
        else:
            return RuleDiscoverer(pool=list()).create_start_tuples(n, X, y)


    # ------------- step() --------------


    def test_step_not_implemented(self):
        """
        Tests the method RuleDiscoverer.step().

        Verifies that NotImplementedError is raised (independent of the parameters).
        """
        self.assertRaises(NotImplementedError, RuleDiscoverer(pool=list()).step, X=None, y=None)


    # ------------- draw_examples_from_data() --------------


    def test_draw_examples_from_data(self):
        """
        Tests the method RuleDiscoverer.draw_examples_from_data().

        Checks that exactly n classifiers are returned by this function.
        """
        x_len, n = (10, 5)
        self.assertEqual(n, len(self.setup_test(x_len=x_len, n=n, start_points='d', elitist=None)))


    # ------------- elitist_unmatched() --------------


    def test_elitist_unmatched_no_solution_optimizer(self):
        """
        Tests the method RuleDiscoverer.elitist_unmatched().

        Checks that this function uses the 'draw examples from data'
        strategy if no solution optimizer is passed.
        """
        x_len, n = (10, 5)
        pool = self.setup_test(x_len=x_len, n=n, start_points='u', elitist=None)
        self.assertGreater(len(pool), 0)


    def test_elitist_unmatched(self):
        """
        Tests the method RuleDiscoverer.elitist_unmatched().

        Checks that this function creates n random
        classifiers from the points unmatched by the
        classifiers used by the elitist.
        """
        x_len, n = (10, 5)

        # Start the ClassifierPool
        pool = list()
        for i in range(x_len):
            classifier = Classifier.random_cl(1)
            classifier.error = 0.2
            classifier.experience = 0
            pool.append(classifier)

        # Get classifiers' intervals
        individual = Individual(np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype='bool'), pool)
        classifier_tuples = self.setup_test(x_len=x_len, n=n, start_points='u', elitist=individual)

        self.assertEqual(len(classifier_tuples), 5)


    def test_elitist_unmatched_n_too_big(self):
        """
        Tests the method RuleDiscoverer.elitist_unmatched().

        If the 'n' is greater than the number of unmatched
        samples, then return only 'len(unmatched_points)'
        samples.
        """
        x_len, n = (10, 1000)

        # Start the ClassifierPool
        pool = list()
        for i in range(x_len):
            classifier = Classifier.random_cl(1)
            classifier.error = 0.2
            classifier.experience = 0
            pool.append(classifier)

        # Create Individual that matches all classifier but one
        individual = Individual(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype='bool'), pool)

        pool = self.setup_test(x_len=x_len, n=n, start_points='u', elitist=individual)
        self.assertEqual(len(pool), x_len)


    # ------------- split_interval() --------------

    def test_split_interval_in_two(self):
        """
        Tests the method RuleDiscoverer.split_interval()

        Checks that the interval [10, 30] is correctly
        divided in 2
        """
        self.assertTrue(np.allclose(RuleDiscoverer(pool=list()).split_interval([10, 30], n=2), [[10, 20], [20, 30]]))


    def test_split_interval_in_four(self):
        """
        Tests the method RuleDiscoverer.split_interval()

        Checks that the interval [10, 30] is correctly
        divided in 4
        """
        self.assertTrue(np.allclose(RuleDiscoverer(pool=list()).split_interval([10, 30], n=4), [[10, 15], [15, 20], [20, 25], [25, 30]]))


    def test_split_interval_negative_numbers(self):
        """
        Tests the method RuleDiscoverer.split_interval()

        Checks the split with negative values.
        """
        self.assertTrue(np.allclose(RuleDiscoverer(pool=list()).split_interval([-30, 0], n=3), [[-30, -20], [-20, -10], [-10, 0]]))


    def test_split_interval_round_zero(self):
        """
        Tests the method RuleDiscoverer.split_interval()

        Checks the split with both negative and positive values.
        """
        self.assertTrue(np.allclose(RuleDiscoverer(pool=list()).split_interval([-20, 20], n=3), [  [-20, -6.66666667],
                                                                                        [-6.66666667, 6.66666667],
                                                                                        [6.66666667, 20]
                                                                                     ]))


    # ------------- elitist_complement() --------------


    def test_elitist_complement_no_solution_optimizer(self):
        """
        Tests the method RuleDiscoverer.elitist_complement().

        Checks that this function uses the 'draw examples from data'
        strategy instead of crashing.
        """
        x_len, n = (10, 5)
        pool = self.setup_test(x_len=x_len, n=n, start_points='c', elitist=None)
        self.assertGreater(len(pool), 0)


    def test_elitist_complement_right_configurations(self):
        """
        Tests the method RuleDiscoverer.create_start_tuples().

        Check that the method 'elitist_complement' is properly
        called, when start_points='c' and
        elitist there is a solution optimizer.
        """
        x_len, n = (10, 4)

        # Start the ClassifierPool
        pool = list()
        for i in range(x_len):
            classifier = Classifier.random_cl(1)
            classifier.error = 0.2
            classifier.experience = 0
            pool.append(classifier)

        # Create Individual that matches all classifier but one
        individual = Individual(np.array([1, 1, 0, 1, 1, 1, 0, 1, 0, 0], dtype='bool'), pool)

        self.assertEqual(len(self.setup_test(x_len=x_len, n=n, start_points='c', elitist=individual)),
                         (n * 2) * np.count_nonzero(individual.genome))

    def test_elitist_complement_array(self):
        """
        Tests the method RuleDiscoverer.elitist_complement().

        Checks that the method 'elitist_complement' is properly
        calculating the array used to create the classifiers.
        """
        x_len, n = (2, 2)

        # Start the ClassifierPool
        pool = list()
        for i in range(x_len):
            classifier = Classifier.random_cl(1)
            classifier.error = 0.2
            classifier.experience = 0
            pool.append(classifier)

        # Create SolutionOptimizer with elitist
        X, y = TestsSupport.generate_input(x_len)
        elitist = Individual(np.array([1, 1], dtype='bool'), pool)
        solution_opt = ES_1plus1(X, y, list(), elitist)

        # Create expected result array
        expectations = np.ndarray((x_len, 1, 4, 2))
        for i in range(x_len):
            cl = pool[i]
            lower_interval = [-1, cl.lowerBounds[0]]
            upper_interval = [cl.upperBounds[0], 1]

            np.concatenate((RuleDiscoverer(pool=list()).split_interval(lower_interval, n),
                            RuleDiscoverer(pool=list()).split_interval(upper_interval, n)),
                            out=expectations[i][0])

        # Create output from method (ignore the classfiers)
        _, out = RuleDiscoverer(pool=list(), solution_optimizer=solution_opt).elitist_complement(n=n, X=X, y=y)

        # Compare
        self.assertTrue(np.allclose(expectations, out))
        mu, lmbd = (15, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, steps_per_step=4)
        X, y = TestsSupport.generate_input(mu)

        optimizer = RuleDiscoverer(pool=[])
        with self.assertRaises(NotImplementedError):
            optimizer.step(X, y)


    # ------------- extract_classifier_attributes() --------------


    # ------------- create_sigmas() --------------


    def test_create_sigmas_xdim_one(self):
        optimizer = RuleDiscoverer(pool=[])
        self.assertTrue(0 <= optimizer.create_sigmas(1) <= 1)


    def test_create_sigmas_xdim_five(self):
        x_dim = 5
        optimizer = RuleDiscoverer(pool=[])
        sigmas = optimizer.create_sigmas(x_dim)
        self.assertTrue((sigmas >= 0).all() and (sigmas <= 1).all())


    # ------------- select_best_classifiers() --------------


    def test_select_best_classifiers(self):
        mu, lmbd = (5, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, steps_per_step=4)
        X, y = TestsSupport.generate_input(mu + 10)
        cls = TestsSupport.mock_classifiers(mu + 10)
        optimizer = RuleDiscoverer(pool=[])
        cls_tuples = list()

        for i in range(mu + 10):
            cl = cls[i]
            cl.fit(X, y)
            cl.error = i
            cls_tuples.append( [cl, optimizer.create_sigmas(1)] )

        self.assertEqual(len(optimizer.select_best_classifiers(cls_tuples, mu)), mu)


    def test_select_best_classifiers_mu_equal_len(self):
        mu, lmbd = (15, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, steps_per_step=4)
        X, y = TestsSupport.generate_input(mu)
        cls = TestsSupport.mock_classifiers(mu)
        optimizer = RuleDiscoverer(pool=[])
        cls_tuples = list()

        for i in range(mu):
            cl = cls[i]
            cl.fit(X, y)
            cl.error = i
            cls_tuples.append( [cl, optimizer.create_sigmas(1)] )

        self.assertEqual(optimizer.select_best_classifiers(cls_tuples, mu), cls_tuples)


    def test_select_best_classifiers_mu_greater_than_len(self):
        mu, lmbd = (10, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, steps_per_step=4)
        X, y = TestsSupport.generate_input(mu)
        cls = TestsSupport.mock_classifiers(mu)
        optimizer = RuleDiscoverer(pool=[])
        cls_tuples = list()

        for i in range(mu):
            cl = cls[i]
            cl.fit(X, y)
            cl.error = i
            cls_tuples.append( [cl, optimizer.create_sigmas(1)] )

        with self.assertRaises(ValueError) as cm:
            optimizer.select_best_classifiers(cls_tuples, mu + 5)
        self.assertEqual('kth(=14) out of bounds (10)', str(cm.exception))




