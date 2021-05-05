from suprb2.config import Config
from suprb2.pool import ClassifierPool
from suprb2.utilities import Utilities
from suprb2.solutions import ES_1plus1
from suprb2.classifier import Classifier
from test.tests_support import TestsSupport
from suprb2.discovery import RuleDiscoverer

import unittest
import numpy as np
from mock import patch, Mock

class TestRuleDiscoverer(unittest.TestCase):
    """
    This module test all methods from RuleDiscoverer
    """


    def setUp(self):
        """
        Resets the Classifier Pool for the next test.
        """
        ClassifierPool().classifiers = list()
        Config().__init__()


    def setup_test(self, x_len, n, start_points, solution_opt):
        TestsSupport.set_rule_discovery_configs(start_points=start_points)
        X, y = TestsSupport.generate_input(x_len)
        return RuleDiscoverer().create_start_points(n, X, y, solution_opt)


    def test_draw_mu_examples_from_data(self):
        """
        Tests the method RuleDiscoverer.draw_mu_examples_from_data().

        Checks that exactly mu classifiers are returned by this function.
        """
        x_len, n = (10, 5)
        self.assertEqual(n, len(self.setup_test(x_len=x_len, n=n, start_points=None, solution_opt=None)))


    def test_elitist_unmatched_raise_error(self):
        """
        Tests the method RuleDiscoverer.elitist_unmatched().

        Checks that this function raises AttributeError:
        'NoneType' object has no attribute 'get_elitist'
        when start_point='elitist_unmatched', solution_opt=None
        """
        x_len, n = (10, 5)
        self.assertRaises(AttributeError, self.setup_test, x_len=x_len, n=n, start_points='elitist_unmatched', solution_opt=None)


    def test_elitist_compliment_raise_error(self):
        """
        Tests the method RuleDiscoverer.elitist_compliment().

        Checks that this function raises AttributeError:
        'NoneType' object has no attribute 'get_elitist'
        when start_point='elitist_compliment', solution_opt=None
        """
        x_len, n = (10, 5)
        self.assertRaises(AttributeError, self.setup_test, x_len=x_len, n=n, start_points='elitist_compliment', solution_opt=None)
