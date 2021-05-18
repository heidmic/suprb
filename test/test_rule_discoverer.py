from suprb2.discovery import RuleDiscoverer
from test.tests_support import TestsSupport

import unittest

class TestDiscoveryRuleDiscoverer(unittest.TestCase):
    """
    This module test all methods from RuleDiscoverer
    """


    # ------------- step() --------------


    def test_step_not_implemented(self):
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




