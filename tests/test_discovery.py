from suprb2.config import Config
from suprb2.pool import ClassifierPool
from suprb2.classifier import Classifier
from tests.test_support import TestSupport
from suprb2.discovery import RuleDiscoverer

import unittest
import numpy as np

class TestDiscovery(unittest.TestCase):
    """
    This module test all methods from RuleDiscoverer
    """


    def setUp(self):
        """
        Resets the Classifier Pool for the next test.
        """
        ClassifierPool().classifiers = list()
        Config().__init__()

    # ------------- step() --------------


    def test_step_mu_equals_lambda_comma(self):
        """
        Tests the method RuleDiscoverer.step().

        With seed = 1, just one of the children have
        errors good enough to be added to the pool.
        Since we are using the ',' replacement, this
        lonly child will replace their parents, and
        we will have a population of 1
        """
        mu, lmbd = (10, 10)
        TestSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=1)
        X, y = TestSupport.initiate_pool(mu, 1)

        optimizer = RuleDiscoverer()
        optimizer.step(X, y)
        self.assertEqual(len(ClassifierPool().classifiers), 1)


    def test_step_mu_equals_lambda_plus(self):
        """
        Tests the method RuleDiscoverer.step().

        With seed = 2, just 2 of the children have
        errors good enough to be added to the pool.
        Since we are using the '+' replacement, this
        lonly child will be added with their parents
        to the next generation (with size 10 + 2 = 12)
        """
        mu, lmbd = (10, 10)
        TestSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement='+', steps_per_step=1)
        X, y = TestSupport.initiate_pool(mu, 2)

        optimizer = RuleDiscoverer()
        optimizer.step(X, y)
        self.assertEqual(len(ClassifierPool().classifiers), 12)


    def test_step_mu_bigger_than_lambda_comma(self):
        """
        Tests the method RuleDiscoverer.step().

        With seed = 1, just one of the children have
        errors good enough to be added to the pool.
        Since we are using the ',' replacement, this
        lonly child will replace their parents, and
        we will have a population of 1
        """
        mu, lmbd = (15, 10)
        TestSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=1)
        X, y = TestSupport.initiate_pool(mu, 1)

        optimizer = RuleDiscoverer()
        optimizer.step(X, y)
        self.assertEqual(len(ClassifierPool().classifiers), 1)


    def test_step_mu_bigger_than_lambda_plus(self):
        """
        Tests the method RuleDiscoverer.step().

        With seed = 1, just one of the children have
        errors good enough to be added to the pool.
        Since we are using the '+' replacement, this
        lonly child will be added with their parents
        to the next generation (with size 15 + 1 = 16)
        """
        mu, lmbd = (15, 10)
        TestSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement='+', steps_per_step=1)
        X, y = TestSupport.initiate_pool(mu, 1)

        optimizer = RuleDiscoverer()
        optimizer.step(X, y)
        self.assertEqual(len(ClassifierPool().classifiers), 16)


    def test_step_mu_smaller_than_lambda_comma(self):
        """
        Tests the method RuleDiscoverer.step().

        With seed = 2, just one of the children have
        errors good enough to be added to the pool.
        Since we are using the ',' replacement, this
        lonly child will replace their parents, and
        we will have a population of 1
        """
        mu, lmbd = (10, 15)
        TestSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=1)
        X, y = TestSupport.initiate_pool(mu, 2)

        optimizer = RuleDiscoverer()
        optimizer.step(X, y)
        self.assertGreaterEqual(len(ClassifierPool().classifiers), 1)


    def test_step_mu_smaller_than_lambda_plus(self):
        """
        Tests the method RuleDiscoverer.step().

        With seed = 1, just one of the children have
        errors good enough to be added to the pool.
        Since we are using the '+' replacement, this
        lonly child will be added with their parents
        to the next generation (with size 10 + 1 = 11)
        """
        mu, lmbd = (10, 15)
        TestSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement='+', steps_per_step=1)
        X, y = TestSupport.initiate_pool(mu, 1)

        optimizer = RuleDiscoverer()
        optimizer.step(X, y)
        self.assertEqual(len(ClassifierPool().classifiers), 11)


    # ------------- remove_parents_from_pool() --------------


    def test_remove_parents_from_pool(self):
        """
        Tests the method RuleDiscoverer.remove_parents_from_pool().

        Checks that parents are no longer in the pool
        after the operation is over.
        """
        mu = 10
        X, y = TestSupport.initiate_pool(mu, 1)
        TestSupport.set_rule_discovery_configs(mu=mu)

        optimizer = RuleDiscoverer()
        parents = optimizer.remove_parents_from_pool()

        self.assertEqual(len(ClassifierPool().classifiers), 0)


    # ------------- recombine() --------------


    def test_recombine_average(self):
        """
        Tests the method RuleDiscoverer.recombine().

        Checks if the average of the classifiers' boundaries
        is propperly calculated.

        child_1.lowerBound = average(random_vater.lowerBound, random_mother.lowerBound)
        child_1.upperBound = average(random_vater.upperBound, random_mother.upperBound)
        """
        TestSupport.set_rule_discovery_configs(recombination='intermediate')
        child = RuleDiscoverer().recombine(TestSupport.mock_specific_classifiers([ [2, 2, 0], [4, 2, 0], [2, 4, 0], [4, 4, 0] ]))[0]
        self.assertIn(child.lowerBounds, [2, 3, 4])
        self.assertIn(child.upperBounds, [2, 3, 4])


    def test_recombine_default_strategy(self):
        """
        Tests the method RuleDiscoverer.recombine().

        Checks if no recombination method is configured, then
        use a default strategy (just copy one of the parents).
        This test do not verify the integrity of the deepcopy,
        instead it just checks that the generated child is a
        Classifier.
        """
        parents = TestSupport.mock_classifiers(10)
        child = RuleDiscoverer().recombine(parents)[0]
        self.assertIsInstance(child, Classifier)


    # ------------- mutate_and_fit() --------------


    def test_mutate_and_fit(self):
        """
        Tests the method RuleDiscoverer.mutate_and_fit().

        Checks if the lower and upper bounds were slightly
        ( <= (u - l) / 10 ) changed and if there is an
        error present.
        """
        n = 2
        classifiers = TestSupport.mock_specific_classifiers([ [[5], [10], None], [[10], [5], None] ])
        X, y = TestSupport.generate_input(n)
        RuleDiscoverer().mutate_and_fit(classifiers, X, y)
        self.assertLessEqual(classifiers[0].lowerBounds, 5 + (10 - 5 / 10))


    # ------------- replace() --------------


    def test_replace_plus(self):
        """
        Tests the method RuleDiscoverer.replace().

        Chaecks that the + replacement operator
        is propperly returning both parents and
        children.
        """
        TestSupport.set_rule_discovery_configs(replacement='+')
        parents = TestSupport.mock_classifiers(5)
        children = TestSupport.mock_classifiers(3)

        array_concatenation = np.concatenate((children, parents))
        replacement_array = RuleDiscoverer().replace(parents, children)
        self.assertIsNone(np.testing.assert_array_equal(replacement_array, array_concatenation))


if __name__ == '__main__':
    unittest.main()
