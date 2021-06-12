from suprb2.utilities import Utilities
from suprb2.solutions import ES_1plus1
from suprb2.discovery import ES_MuLambd
from suprb2.classifier import Classifier
from test.tests_support import TestsSupport
from sklearn.linear_model import LinearRegression

import unittest
import numpy as np
from mock import patch

class TestDiscoveryES_MuLambd(unittest.TestCase):
    """
    This module test all methods from ES_MuLambd
    """


    def assertAlmostIn(self, member, container, atol=0):
        for element in container:
            if np.isclose(member, element, atol=atol):
                return
        raise AssertionError(f'{member} is not in {container}')


    def step_test(self, mu):
        X, y = TestsSupport.generate_input(mu)
        # sol_opt = ES_1plus1(X, y)
        rule_disc = ES_MuLambd()
        rule_disc.step(X, y)


    # ------------- step() --------------


    def test_step_multiple_steps_plus(self):
        """
        Tests the method ES_MuLambd.step().

        Each step, we add the best 'mu' classifiers.
        After 4 steps_per_step, our population will have
        the best 'mu' classifiers.
        """
        mu, lmbd = (15, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement='+', steps_per_step=4, recombination='i')
        X, y = TestsSupport.generate_input(mu)

        optimizer = ES_MuLambd(pool=[])
        optimizer.step(X, y)
        self.assertEqual(len(optimizer.pool), mu)

    @patch.object(Classifier, 'get_weighted_error', return_value=float('-inf'))
    @patch.object(Utilities, 'default_error', return_value=float('inf'))
    def test_step_multiple_steps_comma(self, mock_weighted_error, mock_default_error):
        """
        Tests the method ES_MuLambd.step().

        We create only lmbd (from the initial mu classifiers)
        and each step, we change these classifiers.
        In the end of 4 steps_per_step, we will have the
        best 'mu' classifiers in the pool.
        """
        mu, lmbd = (15, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=4, recombination='i')
        X, y = TestsSupport.generate_input(mu)

        optimizer = ES_MuLambd(pool=[])
        optimizer.step(X, y)
        self.assertEqual(len(optimizer.pool), mu)


    def test_step_mu_bigger_than_population(self):
        """
        Tests the method ES_MuLambd.step().

        If mu is bigger than the population's size, then raise
        Exception (ValueError).
        """
        mu, lmbd = (15, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=4, recombination='i')
        X, y = TestsSupport.generate_input(mu - 5)

        optimizer = ES_MuLambd(pool=[])
        self.assertRaises(ValueError, optimizer.step, X, y)


    def test_step_lambd_zero_comma(self):
        """
        Tests the method ES_MuLambd.step().

        When lmbd is zero and replacement is ',',
        then raise IndexError (it doesn't make any sense to allow this).
        """
        mu, lmbd = (15, 0)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=4, recombination='i')
        X, y = TestsSupport.generate_input(mu)

        optimizer = ES_MuLambd(pool=[])
        self.assertRaises(IndexError, optimizer.step, X, y)


    def test_step_lambd_zero_plus(self):
        """
        Tests the method ES_MuLambd.step().

        When lmbd is zero and replacement is '+',
        then only the initial mu classifiers are
        added to the pool.
        """
        mu, lmbd = (15, 0)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement='+', steps_per_step=4, recombination='i')
        X, y = TestsSupport.generate_input(mu)

        optimizer = ES_MuLambd(pool=[])
        optimizer.step(X, y)
        self.assertEqual(len(optimizer.pool), mu)


    def test_step_mu_zero(self):
        """
        Tests the method ES_MuLambd.step().

        When mu is zero, then raise IndexError.
        """
        mu, lmbd = (0, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=4, recombination='i')
        X, y = TestsSupport.generate_input(10)

        optimizer = ES_MuLambd(pool=[])
        self.assertRaises(IndexError, optimizer.step, X, y)

        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement='+', steps_per_step=4, recombination='d')
        self.assertRaises(IndexError, optimizer.step, X, y)


    def test_step_no_input(self):
        """
        Tests the method ES_MuLambd.step().

        If our X is empty (no data is given), then raise
        IndexError (indexing on empty array).
        """
        mu, lmbd = (0, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=4, recombination='i')
        X, y = TestsSupport.generate_input(0)

        optimizer = ES_MuLambd(pool=[])

        self.assertRaises(IndexError, optimizer.step, X, y)


    # ------------- recombine() --------------


    def test_recombine_average(self):
        """
        Tests the method ES_MuLambd.recombine().

        Checks if the average of the all classifiers'
        boundaries is propperly calculated.

        child_1.lowerBound = average(random_vater.lowerBound, random_mother.lowerBound)
        child_1.upperBound = average(random_vater.upperBound, random_mother.upperBound)
        child_1.sigmas = average(random_vater.sigmas, random_mother.sigmas)
        """
        TestsSupport.set_rule_discovery_configs(recombination='i', rho=4)

        pool = [    (Classifier([2], [2], None, 1), [0.1]),
                    (Classifier([2], [4], None, 1), [0.2]),
                    (Classifier([4], [2], None, 1), [0.4]),
                    (Classifier([4], [4], None, 1), [0.1])  ]

        rule_disc = ES_MuLambd(pool=[])
        children_tuples = rule_disc.recombine(pool)

        for i in range(len(children_tuples)):
            self.assertIn(children_tuples[i][0].lowerBounds, [2, 3, 4])
            self.assertIn(children_tuples[i][0].upperBounds, [2, 3, 4])
            self.assertAlmostEqual(children_tuples[i][1][0], 0.2)


    def test_recombine_average_rho(self):
        """
        Tests the method ES_MuLambd.recombine().

        Checks if the average of the rho classifiers'
        boundaries is propperly calculated.
        """
        TestsSupport.set_rule_discovery_configs(recombination='i', rho=2)
        pool = [    (Classifier([2], [2], None, 1), [0.1]),
                    (Classifier([2], [4], None, 1), [0.2]),
                    (Classifier([4], [2], None, 1), [0.2]),
                    (Classifier([4], [4], None, 1), [0.1])  ]

        rule_disc = ES_MuLambd(pool=[])
        children_tuples = rule_disc.recombine(pool)

        for i in range(len(children_tuples)):
            self.assertIn(children_tuples[i][0].lowerBounds, [2, 3, 4])
            self.assertIn(children_tuples[i][0].upperBounds, [2, 3, 4])
            self.assertAlmostIn(children_tuples[i][1][0], [0.1, 0.15, 0.2], atol=0.00001)


    def test_recombine_discrete_random_values(self):
        """
        Tests the method ES_MuLambd.recombine().

        Checks if the discrete recombination of the classifiers' boundaries
        is propperly calculated.

        child_1.lowerBound = one_random(parents.lowerBounds)
        child_1.upperBound = one_random(parents.upperBounds)
        child_1.sigmas = one_random(parents.sigmas)
        """
        TestsSupport.set_rule_discovery_configs(recombination='d')
        pool = [    (Classifier([1], [40], None, 1), [0.1]),
                    (Classifier([2], [30], None, 1), [0.2]),
                    (Classifier([3], [20], None, 1), [0.3]),
                    (Classifier([4], [10], None, 1), [0.4])  ]

        rule_disc = ES_MuLambd(pool=[])
        children_tuples = rule_disc.recombine(pool)

        for i in range(len(children_tuples)):
            self.assertIn(children_tuples[i][0].lowerBounds, [1, 2, 3, 4])
            self.assertIn(children_tuples[i][0].upperBounds, [10, 20, 30, 40])
            self.assertAlmostIn(children_tuples[i][1][0], [0.1, 0.2, 0.3, 0.4])


    def test_recombine_discrete_flip_is_working(self):
        """
        Tests the method ES_MuLambd.recombine().

        Checks if the random interval for one boundary
        is flipped (upperBound < lowerBound), that the
        recombination will flip them back.
        """
        TestsSupport.set_rule_discovery_configs(recombination='d')
        pool = [    (Classifier([10], [4], None, 1), [0.1]),
                    (Classifier([20], [3], None, 1), [0.2]),
                    (Classifier([30], [2], None, 1), [0.3]),
                    (Classifier([40], [1], None, 1), [0.4])  ]

        rule_disc = ES_MuLambd(pool=[])
        children_tuples = rule_disc.recombine(pool)

        for i in range(len(children_tuples)):
            self.assertIn(children_tuples[i][0].lowerBounds, [1, 2, 3, 4])
            self.assertIn(children_tuples[i][0].upperBounds, [10, 20, 30, 40])
            self.assertAlmostIn(children_tuples[i][1][0], [0.1, 0.2, 0.3, 0.4])


    def test_recombine_default_strategy(self):
        """
        Tests the method ES_MuLambd.recombine().

        Checks if no recombination method is configured, then
        use a default strategy (just copy one of the parents).
        This test do not verify the integrity of the sigmas
        array returned.
        """
        TestsSupport.set_rule_discovery_configs(recombination=None)
        optimizer = ES_MuLambd(pool=[])
        classifiers = TestsSupport.mock_classifiers(10)
        pool = [ (cl, optimizer.create_sigmas(1)) for cl in classifiers ]

        children_tuples = np.array(optimizer.recombine(pool), dtype=object)

        self.assertEqual(children_tuples.shape, (1, 2))
        self.assertIn(children_tuples[0][1], np.array(pool, dtype=object)[:,1])


    # ------------- mutate_and_fit() --------------

    @patch.object(Utilities, 'default_error', return_value=float('inf'))
    def test_mutate_and_fit(self, mock_default_error):
        """
        Tests the method ES_MuLambd.mutate_and_fit().

        Checks if the lower and upper bounds were slightly
        ( <= (u - l) / 10 ) changed and if there is an
        error present.
        """
        n = 2
        optimizer = ES_MuLambd(pool=[])
        classifiers_tuples = [  (Classifier(lowers=[0], uppers=[0.5], local_model=LinearRegression(), degree=1), optimizer.create_sigmas(1)),
                                (Classifier(lowers=[-1], uppers=[0.5], local_model=LinearRegression(), degree=1), optimizer.create_sigmas(1)) ]
        X, y = TestsSupport.generate_input(n)
        mutated_children_tuples = optimizer.mutate_and_fit(classifiers_tuples, X, y)

        self.assertLessEqual(mutated_children_tuples[0][0].lowerBounds, 0.5)


    # ------------- replace() --------------


    def test_replace_plus(self):
        """
        Tests the method ES_MuLambd.replace().

        Chaecks that the + replacement operator
        is propperly returning the sigmas for both
        parents and children.
        """
        TestsSupport.set_rule_discovery_configs(replacement='+')
        optimizer = ES_MuLambd(pool=[])
        parents_tuples = [ (Classifier(lowers=[i], uppers=[i], local_model=None, degree=1), optimizer.create_sigmas(x_dim=1)) for i in range(5) ]
        children_tuples = [ (Classifier(lowers=[i], uppers=[i], local_model=None, degree=1), optimizer.create_sigmas(x_dim=1)) for i in range(5) ]

        array_concatenation = parents_tuples + children_tuples
        replacement_array = optimizer.replace(parents_tuples, children_tuples)

        self.assertCountEqual(replacement_array, array_concatenation)


    def test_replace_comma(self):
        """
        Tests the method ES_MuLambd.replace().

        Chaecks that the , replacement operator
        is propperly returning the children tuples.
        """
        TestsSupport.set_rule_discovery_configs(replacement=',')
        optimizer = ES_MuLambd(pool=[])
        parents_tuples = [ (Classifier(lowers=[i], uppers=[i], local_model=None, degree=1), optimizer.create_sigmas(x_dim=1)) for i in range(5) ]
        children_tuples = [ (Classifier(lowers=[i], uppers=[i], local_model=None, degree=1), optimizer.create_sigmas(x_dim=1)) for i in range(5) ]

        replacement_array = optimizer.replace(parents_tuples, children_tuples)

        self.assertCountEqual(replacement_array, children_tuples)


if __name__ == '__main__':
    unittest.main()
