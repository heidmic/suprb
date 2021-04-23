from suprb2.config import Config
from suprb2.pool import ClassifierPool
from suprb2.classifier import Classifier
from tests.test_support import TestSupport
from suprb2.discovery import ES_MuLambd

import unittest
import numpy as np

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


    def test_step_mu_equals_lambda_comma(self):
        """
        Tests the method ES_MuLambd.step().

        After one step, the population can have max.
        lmbd = 15 classifiers.
        """
        mu, lmbd = (15, 15)
        TestSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=1, recombination='intermediate')
        X, y = TestSupport.initiate_pool(mu, 1)

        optimizer = ES_MuLambd()
        optimizer.step(X, y)
        self.assertEqual(len(ClassifierPool().classifiers), lmbd)


    def test_step_mu_equals_lambda_plus(self):
        """
        Tests the method ES_MuLambd.step().

        After one step, the population can have max.
        mu + lmbd = 20 classifiers.
        """
        mu, lmbd = (10, 10)
        TestSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement='+', steps_per_step=1, recombination='intermediate')
        X, y = TestSupport.initiate_pool(mu, 2)

        optimizer = ES_MuLambd()
        optimizer.step(X, y)
        self.assertLessEqual(len(ClassifierPool().classifiers), mu + lmbd)


    def test_step_mu_bigger_than_lambda_comma(self):
        """
        Tests the method ES_MuLambd.step().

        After one step, the population can have max.
        lmbd = 10 classifiers.
        """
        mu, lmbd = (15, 10)
        TestSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=1, recombination='intermediate')
        X, y = TestSupport.initiate_pool(mu, 1)

        optimizer = ES_MuLambd()
        optimizer.step(X, y)
        self.assertLessEqual(len(ClassifierPool().classifiers), lmbd)


    def test_step_mu_bigger_than_lambda_plus(self):
        """
        Tests the method ES_MuLambd.step().

        After one step, the population can have max.
        mu + lmbd = 25 classifiers.
        """
        mu, lmbd = (15, 10)
        TestSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement='+', steps_per_step=1, recombination='intermediate')
        X, y = TestSupport.initiate_pool(mu, 3)

        optimizer = ES_MuLambd()
        optimizer.step(X, y)
        self.assertLessEqual(len(ClassifierPool().classifiers), mu + lmbd)


    def test_step_mu_smaller_than_lambda_comma(self):
        """
        Tests the method ES_MuLambd.step().

        After one step, the population can have max.
        lmbd = 15 classifiers.
        """
        mu, lmbd = (10, 15)
        TestSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=1, recombination='intermediate')
        X, y = TestSupport.initiate_pool(mu, 2)

        optimizer = ES_MuLambd()
        optimizer.step(X, y)
        self.assertLessEqual(len(ClassifierPool().classifiers), lmbd)


    def test_step_mu_smaller_than_lambda_plus(self):
        """
        Tests the method ES_MuLambd.step().

        After one step, the population can have max.
        mu + lmbd = 25 classifiers.
        """
        mu, lmbd = (10, 15)
        TestSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement='+', steps_per_step=1, recombination='intermediate')
        X, y = TestSupport.initiate_pool(mu, 2)

        optimizer = ES_MuLambd()
        optimizer.step(X, y)
        self.assertLessEqual(len(ClassifierPool().classifiers), mu + lmbd)


    # ------------- select_parents_from_pool() --------------


    def test_select_parents_from_pool(self):
        """
        Tests the method ES_MuLambd.select_parents_from_pool().

        Checks that parents are no longer in the pool
        after the operation is over.
        """
        mu = 10
        X, y = TestSupport.initiate_pool(mu, 1)
        TestSupport.set_rule_discovery_configs(mu=mu)

        optimizer = ES_MuLambd()
        parents = optimizer.select_parents_from_pool()

        self.assertEqual(len(ClassifierPool().classifiers), mu)
        self.assertEqual(parents.size, mu)
        for cl in parents:
            self.assertIn(cl, ClassifierPool().classifiers)


    # ------------- recombine() --------------


    def test_recombine_average(self):
        """
        Tests the method ES_MuLambd.recombine().

        Checks if the average of the classifiers' boundaries
        is propperly calculated.

        child_1.lowerBound = average(random_vater.lowerBound, random_mother.lowerBound)
        child_1.upperBound = average(random_vater.upperBound, random_mother.upperBound)
        """
        TestSupport.set_rule_discovery_configs(recombination='intermediate')
        child = ES_MuLambd().recombine(TestSupport.mock_specific_classifiers([ [2, 2, 0], [4, 2, 0], [2, 4, 0], [4, 4, 0] ]))[0]
        self.assertIn(child.lowerBounds, [2, 3, 4])
        self.assertIn(child.upperBounds, [2, 3, 4])


    def test_recombine_discrete_random_values(self):
        """
        Tests the method ES_MuLambd.recombine().

        Checks if the discrete recombination of the classifiers' boundaries
        is propperly calculated.

        child_1.lowerBound = one_random(parents.lowerBounds)
        child_1.upperBound = one_random(parents.upperBounds)
        """
        TestSupport.set_rule_discovery_configs(recombination='discrete')
        child = ES_MuLambd().recombine(TestSupport.mock_specific_classifiers([ [[1], [40], 0], [[2], [30], 0], [[3], [20], 0], [[4], [10], 0] ]))[0]
        self.assertIn(child.lowerBounds, [1, 2, 3, 4])
        self.assertIn(child.upperBounds, [10, 20, 30, 40])


    def test_recombine_discrete_flip_is_working(self):
        """
        Tests the method ES_MuLambd.recombine().

        Checks if the random interval for one boundary
        is flipped (upperBound < lowerBound), that the
        recombination will flip them back.
        """
        TestSupport.set_rule_discovery_configs(recombination='discrete')
        child = ES_MuLambd().recombine(TestSupport.mock_specific_classifiers([ [[10], [4], 0], [[20], [3], 0], [[30], [2], 0], [[40], [1], 0] ]))[0]
        self.assertIn(child.lowerBounds, [1, 2, 3, 4])
        self.assertIn(child.upperBounds, [10, 20, 30, 40])


    def test_recombine_default_strategy(self):
        """
        Tests the method ES_MuLambd.recombine().

        Checks if no recombination method is configured, then
        use a default strategy (just copy one of the parents).
        This test do not verify the integrity of the deepcopy,
        instead it just checks that the generated child is a
        Classifier.
        """
        parents = TestSupport.mock_classifiers(10)
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
        classifiers = TestSupport.mock_specific_classifiers([ [[5], [10], None], [[10], [5], None] ])
        X, y = TestSupport.generate_input(n)
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
        TestSupport.set_rule_discovery_configs(replacement='+')
        parents = TestSupport.mock_classifiers(5)
        children = TestSupport.mock_classifiers(3)

        array_concatenation = np.concatenate((children, parents))
        replacement_array = ES_MuLambd().replace(parents, children)
        self.assertIsNone(np.testing.assert_array_equal(replacement_array, array_concatenation))


if __name__ == '__main__':
    unittest.main()
