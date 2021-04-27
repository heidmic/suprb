from suprb2.config import Config
from suprb2.pool import ClassifierPool
from suprb2.utilities import Utilities
from suprb2.discovery import ES_MuLambd
from suprb2.classifier import Classifier
from tests.tests_support import TestsSupport

import unittest
import numpy as np
from mock import patch, Mock

class TestDiscoveryES_MuLambd(unittest.TestCase):
    """
    This module test all methods from ES_MuLambd
    """


    def setUp(self):
        """
        Resets the Classifier Pool for the next test.
        """
        ClassifierPool().classifiers = list()
        Config().__init__()


    # ------------- step() --------------


    @patch.object(Classifier, 'get_weighted_error', return_value=float('-inf'))
    @patch.object(Utilities, 'default_error', return_value=0.5)
    def test_step_mu_equals_lambda(self, mock_error, mock_default_error):
        """
        Tests the method ES_MuLambd.step().

        Each step, we add 15 classifiers (errors are mocked).
        After 1 step, our population should have mu + lmbd
        classifiers.
        """
        mu, lmbd = (15, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=1, recombination='intermediate')
        X, y = TestsSupport.initiate_pool(mu, 1)

        optimizer = ES_MuLambd()
        optimizer.step(X, y)
        self.assertEqual(len(ClassifierPool().classifiers), mu + lmbd)


    @patch.object(Classifier, 'get_weighted_error', return_value=float('-inf'))
    @patch.object(Utilities, 'default_error', return_value=0.5)
    def test_step_multiple_steps(self, mock_error, mock_default_error):
        """
        Tests the method ES_MuLambd.step().

        Each step, we add 15 classifiers (errors are mocked).
        After 4 steps, our population should have mu + (lmbd * 4)
        classifiers.
        """
        mu, lmbd = (15, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=4, recombination='intermediate')
        X, y = TestsSupport.initiate_pool(mu, 1)

        optimizer = ES_MuLambd()
        optimizer.step(X, y)
        self.assertEqual(len(ClassifierPool().classifiers), mu + (lmbd * 4))


    @patch.object(Classifier, 'get_weighted_error', return_value=float('-inf'))
    @patch.object(Utilities, 'default_error', return_value=0.5)
    def test_step_mu_bigger_than_population(self, mock_error, mock_default_error):
        """
        Tests the method ES_MuLambd.step().

        If mu is bigger than the population's size, then raise
        Exception (ValueError).
        """
        mu, lmbd = (15, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=4, recombination='intermediate')
        X, y = TestsSupport.initiate_pool(mu - 5, 1)

        optimizer = ES_MuLambd()
        self.assertRaises(ValueError, optimizer.step, X, y)


    @patch.object(Classifier, 'get_weighted_error', return_value=float('-inf'))
    @patch.object(Utilities, 'default_error', return_value=0.5)
    def test_step_lambd_zero(self, mock_error, mock_default_error):
        """
        Tests the method ES_MuLambd.step().

        When lmbd is zero, then nothing is added to the
        population.
        """
        mu, lmbd = (15, 0)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=4, recombination='intermediate')
        X, y = TestsSupport.initiate_pool(mu, 1)

        optimizer = ES_MuLambd()
        optimizer.step(X, y)
        self.assertEqual(len(ClassifierPool().classifiers), mu)


    @patch.object(Classifier, 'get_weighted_error', return_value=float('-inf'))
    @patch.object(Utilities, 'default_error', return_value=0.5)
    def test_step_mu_zero(self, mock_error, mock_default_error):
        """
        Tests the method ES_MuLambd.step().

        When mu is zero, then no classifier is added to the pool.
        """
        mu, lmbd = (0, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=4, recombination='intermediate')
        X, y = TestsSupport.initiate_pool(mu, 1)

        optimizer = ES_MuLambd()
        optimizer.step(X, y)
        self.assertEqual(len(ClassifierPool().classifiers), mu)


    @patch.object(Classifier, 'get_weighted_error', return_value=float('-inf'))
    @patch.object(Utilities, 'default_error', return_value=0.5)
    def test_step_no_input(self, mock_error, mock_default_error):
        """
        Tests the method ES_MuLambd.step().

        If our X is empty (no data is given), then our population will
        remain empty.
        """
        mu, lmbd = (0, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=4, recombination='intermediate')
        X, y = TestsSupport.initiate_pool(0, 1)

        optimizer = ES_MuLambd()
        optimizer.step(X, y)
        self.assertEqual(len(ClassifierPool().classifiers), 0)


    # ------------- recombine() --------------


    def test_recombine_average(self):
        """
        Tests the method ES_MuLambd.recombine().

        Checks if the average of the classifiers' boundaries
        is propperly calculated.

        child_1.lowerBound = average(random_vater.lowerBound, random_mother.lowerBound)
        child_1.upperBound = average(random_vater.upperBound, random_mother.upperBound)
        """
        TestsSupport.set_rule_discovery_configs(recombination='intermediate')
        child = ES_MuLambd().recombine(TestsSupport.mock_specific_classifiers([ [2, 2], [4, 2], [2, 4], [4, 4] ]))[0]
        self.assertIn(child.lowerBounds, [2, 3, 4])
        self.assertIn(child.upperBounds, [2, 3, 4])


    def test_recombine_default_strategy(self):
        """
        Tests the method ES_MuLambd.recombine().

        Checks if no recombination method is configured, then
        use a default strategy (just copy one of the parents).
        This test do not verify the integrity of the deepcopy,
        instead it just checks that the generated child is a
        Classifier.
        """
        parents = TestsSupport.mock_classifiers(10)
        child = ES_MuLambd().recombine(parents)[0]
        self.assertIsInstance(child, Classifier)


    # ------------- mutate_and_fit() --------------


    def test_mutate_and_fit(self):
        """
        Tests the method ES_MuLambd.mutate_and_fit().

        Checks if the lower and upper bounds were slightly
        ( <= (u - l) / 10 ) changed and if there is an
        error present.
        """
        n = 2
        classifiers = TestsSupport.mock_specific_classifiers([ [[5], [10]], [[10], [5]] ])
        X, y = TestsSupport.generate_input(n)
        ES_MuLambd().mutate_and_fit(classifiers, X, y)
        self.assertLessEqual(classifiers[0].lowerBounds, 5 + (10 - 5 / 10))


    # ------------- replace() --------------


    def test_replace_plus(self):
        """
        Tests the method ES_MuLambd.replace().

        Chaecks that the + replacement operator
        is propperly returning both parents and
        children.
        """
        TestsSupport.set_rule_discovery_configs(replacement='+')
        parents = TestsSupport.mock_classifiers(5)
        children = TestsSupport.mock_classifiers(3)

        array_concatenation = np.concatenate((children, parents))
        replacement_array = ES_MuLambd().replace(parents, children)
        self.assertIsNone(np.testing.assert_array_equal(replacement_array, array_concatenation))


if __name__ == '__main__':
    unittest.main()
