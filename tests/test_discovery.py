from tests.examples import Examples
from suprb2.pool import ClassifierPool
from suprb2.classifier import Classifier
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

    # ------------- step() --------------


    def test_step_mu_equals_lambda(self):
        """
        Tests the method RuleDiscoverer.step().

        Test that when mu is equal to lambda, then the population
        will not grow nor sink.
        Classifier will only be added if their error is acceptable,
        that is why all errors are mocked with -1.
        """
        mu, lmbd = (10, 10)
        Examples.set_rule_discovery_configs(mu=mu, lmbd=lmbd, selection='+', steps_per_step=1)
        X, y = Examples.initiate_pool(mu, 1)

        optimizer = RuleDiscoverer()
        optimizer.step(X, y)
        self.assertEqual(len(ClassifierPool().classifiers), mu)


    def test_step_mu_bigger_than_lambda(self):
        """
        Tests the method RuleDiscoverer.step().

        Test that when mu is bigger than lambda, then the population
        will shrink.
        Classifier will only be added if their error is acceptable,
        that is why all errors are mocked with -1.
        """
        mu, lmbd = (15, 10)
        Examples.set_rule_discovery_configs(mu=mu, lmbd=lmbd, selection='+', steps_per_step=1)
        X, y = Examples.initiate_pool(mu, 1)

        optimizer = RuleDiscoverer()
        optimizer.step(X, y)
        self.assertLess(len(ClassifierPool().classifiers), mu)


    def test_step_mu_smaller_than_lambda(self):
        """
        Tests the method RuleDiscoverer.step().

        Test that when mu is equal to lambda, then the population
        will grow.
        Classifier will only be added if their error is acceptable,
        that is why all errors are mocked with -1.
        """
        mu, lmbd = (10, 15)
        Examples.set_rule_discovery_configs(mu=mu, lmbd=lmbd, selection='+', steps_per_step=1)
        X, y = Examples.initiate_pool(mu, 1)

        optimizer = RuleDiscoverer()
        optimizer.step(X, y)
        self.assertGreaterEqual(len(ClassifierPool().classifiers), mu)


    # ------------- remove_parents_from_pool() --------------


    def test_remove_parents_from_pool(self):
        """
        Tests the method RuleDiscoverer.remove_parents_from_pool().

        Checks that parents are no longer in the pool
        after the operation is over.
        """
        mu = 10
        X, y = Examples.initiate_pool(mu, 1)
        Examples.set_rule_discovery_configs(mu=mu)

        optimizer = RuleDiscoverer()
        parents = optimizer.remove_parents_from_pool()

        self.assertEqual(len(ClassifierPool().classifiers), 0)


    # ------------- recombine() --------------


    def test_recombine_average(self):
        """
        Tests the method RuleDiscoverer.recombine().

        Checks if the average of the classifiers' boundaries
        is propperly calculated.

        child.lowerBound = average(parents.lowerBounds)
        child.upperBound = average(parents.upperBounds)
        """
        Examples.set_rule_discovery_configs(recombination='intermediate')
        child = RuleDiscoverer().recombine(Examples.mock_classifiers(10))
        self.assertEquals((child.lowerBounds, child.upperBounds), (5, 5))


    def test_recombine_default_strategy(self):
        """
        Tests the method RuleDiscoverer.recombine().

        Checks if no recombination method is configured, then
        use a default strategy (just copy one of the parents).
        This test do not verify the integrity of the deepcopy,
        instead it just checks that the generated child is a
        Classifier.
        """
        parents = Examples.mock_classifiers(10)
        child = RuleDiscoverer().recombine(parents)
        self.assertIsInstance(child, Classifier)


    # ------------- replace() --------------


    def test_replace_plus(self):
        """
        Tests the method RuleDiscoverer.replace().

        Chaecks that the + replacement operator
        is propperly returning both parents and
        children.
        """
        Examples.set_rule_discovery_configs(replacement='+')
        parents = Examples.mock_classifiers(5)
        children = Examples.mock_classifiers(3)

        array_concatenation = np.concatenate((children, parents))
        replacement_array = RuleDiscoverer().replace(parents, children)
        self.assertIsNone(np.testing.assert_array_equal(replacement_array, array_concatenation))


    # ------------- filter_classifiers() --------------


    def test_filter_classifiers_in_pool(self):
        """
        Tests the method RuleDiscoverer.filter_classifiers().

        Checks that all classifiers filtered have an error
        smaller than the default one.
        """
        X, y = Examples.initiate_pool(10, 1)

        optimizer = RuleDiscoverer()
        filtered_cls = optimizer.filter_classifiers(np.array(ClassifierPool().classifiers), X, y)
        for cl in filtered_cls:
            self.assertTrue(cl.error < optimizer.default_error(y))


    def test_filter_classifiers_that_were_filtered(self):
        """
        Tests the method RuleDiscoverer.filter_classifiers().

        Checks that all classfiers that are not in the
        filtered group have either no error (error == None)
        or error is bigger than the default error.
        """
        X, y = Examples.initiate_pool(10, 1)

        optimizer = RuleDiscoverer()
        filtered_cls = optimizer.filter_classifiers(np.array(ClassifierPool().classifiers), X, y)
        for cl in ClassifierPool().classifiers:
            if cl not in filtered_cls:
                self.assertTrue(cl.error is None or cl.error > optimizer.default_error(y))


    # ------------- best_lambda_classifiers() --------------


    def test_best_lambda_classifiers_sorted(self):
        """
        Tests the method RuleDiscoverer.best_lambda_classifiers().

        Checks that the lambda best classifiers are sorted
        ascending acoording to their errors.
        """
        lmbd = 5
        candidates = Examples.mock_classifiers(10, errors=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        best_lambda_cls = RuleDiscoverer().best_lambda_classifiers(candidates, lmbd)

        self.assertEquals(len(best_lambda_cls), lmbd)
        self.assertTrue(all(best_lambda_cls[i].error <= best_lambda_cls[i + 1].error for i in range(lmbd - 1)))


    def test_best_lambda_classifiers_lowest_errors(self):
        """
        Tests the method RuleDiscoverer.best_lambda_classifiers().

        Checks that the other classifiers have bigger
        errors than any of the lambda classifiers.
        """
        lmbd = 5
        candidates = Examples.mock_classifiers(10, errors=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        best_lambda_cls = RuleDiscoverer().best_lambda_classifiers(candidates, lmbd)

        for cl in candidates:
            if cl not in best_lambda_cls:
                self.assertTrue(cl.error > best_lambda_cls[-1].error)


    def test_best_lambda_classifiers_candidates_smaller_than_lambda(self):
        """
        Tests the method RuleDiscoverer.best_lambda_classifiers().

        If candidates.size < lmbd, then lmbd = candidates.size.
        """
        lmbd = 10
        candidates = Examples.mock_classifiers(5, errors=[0, 1, 2, 3, 4])
        old_size = len(candidates)
        best_lambda_cls = RuleDiscoverer().best_lambda_classifiers(candidates, lmbd)

        self.assertEquals(old_size, len(best_lambda_cls))


if __name__ == '__main__':
    unittest.main()
