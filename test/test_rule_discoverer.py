from suprb2.config import Config
from suprb2.pool import ClassifierPool
from suprb2.utilities import Utilities
from suprb2.solutions import ES_1plus1
from suprb2.individual import Individual
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


    def setup_test(self, x_len, n, start_points, elitist):
        TestsSupport.set_rule_discovery_configs(start_points=start_points)
        X, y = TestsSupport.generate_input(x_len)
        if elitist is not None:
            solution_opt = ES_1plus1(X, y, elitist)
            return RuleDiscoverer().create_start_points(n, X, y, solution_opt)
        else:
            return RuleDiscoverer().create_start_points(n, X, y, None)


    # ------------- step() --------------


    def test_step_not_implemented(self):
        """
        Tests the method RuleDiscoverer.step().

        Verifies that NotImplementedError is raised (independent of the parameters).
        """
        self.assertRaises(NotImplementedError, RuleDiscoverer().step, X=None, y=None, solution_opt=None)


    # ------------- draw_examples_from_data() --------------


    def test_draw_examples_from_data(self):
        """
        Tests the method RuleDiscoverer.draw_examples_from_data().

        Checks that exactly n classifiers are returned by this function.
        """
        x_len, n = (10, 5)
        self.assertEqual(n, len(self.setup_test(x_len=x_len, n=n, start_points=None, elitist=None)))


    # ------------- elitist_unmatched() --------------


    def test_elitist_unmatched_raise_error(self):
        """
        Tests the method RuleDiscoverer.elitist_unmatched().

        Checks that this function raises AttributeError:
        'NoneType' object has no attribute 'get_elitist'
        when start_point='elitist_unmatched', solution_opt=None
        """
        x_len, n = (10, 5)
        self.assertRaises(AttributeError, self.setup_test, x_len=x_len, n=n, start_points='elitist_unmatched', elitist=None)


    def test_elitist_unmatched(self):
        """
        Tests the method RuleDiscoverer.elitist_unmatched().

        Checks that this function creates n copies from the
        classifiers not used by the elitist individual.
        """
        x_len, n = (10, 5)

        # Start the ClassifierPool
        for i in range(x_len):
            classifier = Classifier.random_cl(None, 1)
            classifier.error = 0.2
            classifier.experience = 0
            ClassifierPool().classifiers.append(classifier)

        # Get classifiers' intervals
        individual = Individual(np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype='bool'))
        classifiers_intervals = [[cl.lowerBounds, cl.upperBounds] for cl in individual.get_classifiers(unmatched=True)]

        classifiers = self.setup_test(x_len=x_len, n=n, start_points='elitist_unmatched', elitist=individual)
        for cl in classifiers:
            self.assertIn([cl.lowerBounds, cl.upperBounds], classifiers_intervals)


    def test_elitist_unmatched_n_too_big(self):
        """
        Tests the method RuleDiscoverer.elitist_unmatched().

        Checks that this function raises error if 'n' is bigger
        than the number of classifiers available.
        """
        x_len, n = (10, 5)

        # Start the ClassifierPool
        for i in range(x_len):
            classifier = Classifier.random_cl(None, 1)
            classifier.error = 0.2
            classifier.experience = 0
            ClassifierPool().classifiers.append(classifier)

        # Create Individual that matches all classifier but one
        individual = Individual(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype='bool'))

        self.assertRaises(ValueError, self.setup_test, x_len=x_len, n=n, start_points='elitist_unmatched', elitist=individual)


    # ------------- split_interval() --------------

    def test_split_interval_in_two(self):
        """
        Tests the method RuleDiscoverer.split_interval()

        Checks that the interval [10, 30] is correctly
        divided in 2
        """
        self.assertTrue(np.allclose(RuleDiscoverer().split_interval([10, 30], n=2), [[10, 20], [20, 30]]))


    def test_split_interval_in_four(self):
        """
        Tests the method RuleDiscoverer.split_interval()

        Checks that the interval [10, 30] is correctly
        divided in 4
        """
        self.assertTrue(np.allclose(RuleDiscoverer().split_interval([10, 30], n=4), [[10, 15], [15, 20], [20, 25], [25, 30]]))


    def test_split_interval_negative_numbers(self):
        """
        Tests the method RuleDiscoverer.split_interval()

        Checks the split with negative values.
        """
        self.assertTrue(np.allclose(RuleDiscoverer().split_interval([-30, 0], n=3), [[-30, -20], [-20, -10], [-10, 0]]))


    def test_split_interval_round_zero(self):
        """
        Tests the method RuleDiscoverer.split_interval()

        Checks the split with both negative and positive values.
        """
        self.assertTrue(np.allclose(RuleDiscoverer().split_interval([-20, 20], n=3), [  [-20, -6.66666667],
                                                                                        [-6.66666667, 6.66666667],
                                                                                        [6.66666667, 20]
                                                                                     ]))


    # ------------- elitist_complement() --------------


    def test_elitist_complement_raise_error(self):
        """
        Tests the method RuleDiscoverer.elitist_complement().

        Checks that this function raises AttributeError:
        'NoneType' object has no attribute 'get_elitist'
        when start_point='elitist_complement', solution_opt=None
        """
        x_len, n = (10, 5)
        self.assertRaises(AttributeError, self.setup_test, x_len=x_len, n=n, start_points='elitist_complement', elitist=None)


    def test_elitist_complement_right_configurations(self):
        """
        Tests the method RuleDiscoverer.create_start_points().

        Check that the method 'elitist_complement' is properly
        called, when start_points='elitist_complement' and
        elitist there is a solution optimizer.
        """
        x_len, n = (10, 4)

        # Start the ClassifierPool
        for i in range(x_len):
            classifier = Classifier.random_cl(None, 1)
            classifier.error = 0.2
            classifier.experience = 0
            ClassifierPool().classifiers.append(classifier)

        # Create Individual that matches all classifier but one
        individual = Individual(np.array([1, 1, 0, 1, 1, 1, 0, 1, 0, 0], dtype='bool'))

        self.assertEqual(len(self.setup_test(x_len=x_len, n=n, start_points='elitist_complement', elitist=individual)),
                         (n * 2) * np.count_nonzero(individual.genome))

    def test_elitist_complement_array(self):
        """
        Tests the method RuleDiscoverer.elitist_complement().

        Checks that the method 'elitist_complement' is properly
        calculating the array used to create the classifiers.
        """
        x_len, n = (2, 2)

        # Start the ClassifierPool
        for i in range(x_len):
            classifier = Classifier.random_cl(None, 1)
            classifier.error = 0.2
            classifier.experience = 0
            ClassifierPool().classifiers.append(classifier)

        # Create SolutionOptimizer with elitist
        X, y = TestsSupport.generate_input(x_len)
        elitist = Individual(np.array([1, 1], dtype='bool'))
        solution_opt = ES_1plus1(X, y, elitist)

        # Create expected result array
        expectations = np.ndarray((x_len, 1, 4, 2))
        for i in range(x_len):
            cl = ClassifierPool().classifiers[i]
            lower_interval = [-1, cl.lowerBounds[0]]
            upper_interval = [cl.upperBounds[0], 1]

            np.concatenate((RuleDiscoverer().split_interval(lower_interval, n),
                            RuleDiscoverer().split_interval(upper_interval, n)),
                            out=expectations[i][0])

        # Create output from method (ignore the classfiers)
        _, out = RuleDiscoverer().elitist_complement(n=n, X=X, y=y, solution_opt=solution_opt)

        # Compare
        self.assertTrue(np.allclose(expectations, out))
