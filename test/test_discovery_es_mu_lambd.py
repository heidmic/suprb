from suprb2.config import Config
from suprb2.utilities import Utilities
from suprb2.discovery import ES_MuLambd
from suprb2.classifier import Classifier
from test.tests_support import TestsSupport

import unittest
import numpy as np
from mock import patch, Mock

class TestDiscoveryES_MuLambd(unittest.TestCase):
    """
    This module test all methods from ES_MuLambd
    """


    def assertAlmostIn(self, member, container):
        for element in container:
            if np.isclose(member, element):
                return
        raise AssertionError(f'{member} is not in {container}')


    # ------------- step() --------------


    @patch.object(Classifier, 'get_weighted_error', return_value=float('-inf'))
    @patch.object(Utilities, 'default_error', return_value=0.5)
    def test_step_multiple_steps_plus(self, mock_error, mock_default_error):
        """
        Tests the method ES_MuLambd.step().

        Each step, we add 15 classifiers (before the first step,
        we add mu classifiers).
        After 4 steps, our population should have mu + (lmbd * 4)
        classifiers.
        """
        mu, lmbd = (15, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement='+', steps_per_step=4, recombination='intermediate')
        X, y = TestsSupport.generate_input(mu)

        optimizer = ES_MuLambd(pool=[])
        optimizer.step(X, y)
        self.assertEqual(len(optimizer.pool), mu + (lmbd * 4))


    @patch.object(Classifier, 'get_weighted_error', return_value=float('-inf'))
    @patch.object(Utilities, 'default_error', return_value=0.5)
    def test_step_multiple_steps_comma(self, mock_error, mock_default_error):
        """
        Tests the method ES_MuLambd.step().

        We create only lmbd (from the initial mu classifiers)
        and each step, we change these classifiers.
        In the end of 4 steps_per_step, we will have lmbd
        classifiers in the pool.
        """
        mu, lmbd = (15, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=4, recombination='intermediate')
        X, y = TestsSupport.generate_input(mu)

        optimizer = ES_MuLambd(pool=[])
        optimizer.step(X, y)
        self.assertEqual(len(optimizer.pool), lmbd)


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
        X, y = TestsSupport.generate_input(mu - 5)

        optimizer = ES_MuLambd(pool=[])
        self.assertRaises(ValueError, optimizer.step, X, y)


    @patch.object(Classifier, 'get_weighted_error', return_value=float('-inf'))
    @patch.object(Utilities, 'default_error', return_value=0.5)
    def test_step_lambd_zero_comma(self, mock_error, mock_default_error):
        """
        Tests the method ES_MuLambd.step().

        When lmbd is zero and replacement is ',',
        then no classifier is added to the pool.
        """
        mu, lmbd = (15, 0)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=4, recombination='intermediate')
        X, y = TestsSupport.generate_input(mu)

        optimizer = ES_MuLambd(pool=[])
        optimizer.step(X, y)
        self.assertEqual(len(optimizer.pool), 0)


    @patch.object(Classifier, 'get_weighted_error', return_value=float('-inf'))
    @patch.object(Utilities, 'default_error', return_value=0.5)
    def test_step_lambd_zero_plus(self, mock_error, mock_default_error):
        """
        Tests the method ES_MuLambd.step().

        When lmbd is zero and replacement is '+',
        then only the initial mu classifiers are
        added to the pool.
        """
        mu, lmbd = (15, 0)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement='+', steps_per_step=4, recombination='intermediate')
        X, y = TestsSupport.generate_input(mu)

        optimizer = ES_MuLambd(pool=[])
        optimizer.step(X, y)
        self.assertEqual(len(optimizer.pool), mu)


    @patch.object(Classifier, 'get_weighted_error', return_value=float('-inf'))
    @patch.object(Utilities, 'default_error', return_value=0.5)
    def test_step_mu_zero(self, mock_error, mock_default_error):
        """
        Tests the method ES_MuLambd.step().

        When mu is zero, then no classifier is added to the pool
        (independent of the replacement).
        """
        mu, lmbd = (0, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=4, recombination='intermediate')
        X, y = TestsSupport.generate_input(mu)

        optimizer = ES_MuLambd(pool=[])
        optimizer.step(X, y)
        self.assertEqual(len(optimizer.pool), 0)

        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement='+', steps_per_step=4, recombination='intermediate')
        optimizer.step(X, y)
        self.assertEqual(len(optimizer.pool), 0)



    @patch.object(Classifier, 'get_weighted_error', return_value=float('-inf'))
    @patch.object(Utilities, 'default_error', return_value=0.5)
    def test_step_no_input(self, mock_error, mock_default_error):
        """
        Tests the method ES_MuLambd.step().

        If our X is empty (no data is given), then our population will
        remain empty.
        """
        mu, lmbd = (0, 15)
        TestsSupport.set_rule_discovery_configs(mu=mu, lmbd=lmbd, replacement=',', steps_per_step=4, recombination='intermediate', simga=0.2)
        X, y = TestsSupport.generate_input(0)

        optimizer = ES_MuLambd(pool=[])
        optimizer.step(X, y)
        self.assertEqual(len(optimizer.pool), 0)


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
        TestsSupport.set_rule_discovery_configs(recombination='intermediate', rho=4)

        classifiers = [ Classifier([2], [2], None, 1),
                        Classifier([2], [4], None, 1),
                        Classifier([4], [2], None, 1),
                        Classifier([4], [4], None, 1)]
        sigmas      = [ [0.1],
                        [0.2],
                        [0.4],
                        [0.1]]

        rule_disc = ES_MuLambd(pool=classifiers)
        rule_disc.register_classifier_sigmas(classifiers, sigmas)
        children, children_sigmas = rule_disc.recombine(classifiers, sigmas)

        self.assertAlmostEqual(len(children), len(children_sigmas))
        for i in range(len(children)):
            self.assertIn(children[i].lowerBounds, [2, 3, 4])
            self.assertIn(children[i].upperBounds, [2, 3, 4])
            self.assertAlmostIn(children_sigmas[i], [0.2])


    def test_recombine_rho_average(self):
        """
        Tests the method ES_MuLambd.recombine().

        Checks if the average of the rho classifiers'
        boundaries is propperly calculated.
        """
        TestsSupport.set_rule_discovery_configs(recombination='intermediate', rho=2)
        classifiers = [ Classifier([2], [2], None, 1),
                        Classifier([2], [4], None, 1),
                        Classifier([4], [2], None, 1),
                        Classifier([4], [4], None, 1)]
        sigmas      = [ [0.1],
                        [0.2],
                        [0.2],
                        [0.1]]

        rule_disc = ES_MuLambd(classifiers)
        rule_disc.register_classifier_sigmas(classifiers, sigmas=sigmas)
        children, children_sigmas = rule_disc.recombine(classifiers, sigmas=sigmas)

        self.assertEqual(len(children), len(children_sigmas))
        for i in range(len(children)):
            self.assertIn(children[i].lowerBounds, [2, 3, 4])
            self.assertIn(children[i].upperBounds, [2, 3, 4])
            self.assertAlmostIn(children_sigmas[i], [0.1, 0.15, 0.2])


    def test_recombine_discrete_random_values(self):
        """
        Tests the method ES_MuLambd.recombine().

        Checks if the discrete recombination of the classifiers' boundaries
        is propperly calculated.

        child_1.lowerBound = one_random(parents.lowerBounds)
        child_1.upperBound = one_random(parents.upperBounds)
        child_1.sigmas = one_random(parents.sigmas)
        """
        TestsSupport.set_rule_discovery_configs(recombination='discrete')
        classifiers = [ Classifier([1], [40], None, 1),
                        Classifier([2], [30], None, 1),
                        Classifier([3], [20], None, 1),
                        Classifier([4], [10], None, 1)]
        sigmas      = [ [0.1],
                        [0.2],
                        [0.3],
                        [0.4]]

        rule_disc = ES_MuLambd(classifiers)
        rule_disc.register_classifier_sigmas(classifiers, sigmas=sigmas)
        children, children_sigmas = rule_disc.recombine(classifiers, sigmas=sigmas)

        self.assertEqual(len(children), len(children_sigmas))
        for i in range(len(children)):
            self.assertIn(children[i].lowerBounds, [1, 2, 3, 4])
            self.assertIn(children[i].upperBounds, [10, 20, 30, 40])
            self.assertAlmostIn(children_sigmas[i], [0.1, 0.2, 0.3, 0.4])


    def test_recombine_discrete_flip_is_working(self):
        """
        Tests the method ES_MuLambd.recombine().

        Checks if the random interval for one boundary
        is flipped (upperBound < lowerBound), that the
        recombination will flip them back.
        """
        TestsSupport.set_rule_discovery_configs(recombination='discrete')
        classifiers = [ Classifier([10], [4], None, 1),
                        Classifier([20], [3], None, 1),
                        Classifier([30], [2], None, 1),
                        Classifier([40], [1], None, 1)]
        sigmas      = [ [0.1],
                        [0.2],
                        [0.3],
                        [0.4]]

        rule_disc = ES_MuLambd(classifiers)
        rule_disc.register_classifier_sigmas(classifiers, sigmas=sigmas)
        children, children_sigmas = rule_disc.recombine(classifiers, sigmas=sigmas)

        self.assertEqual(len(children), len(children_sigmas))
        for i in range(len(children)):
            self.assertIn(children[i].lowerBounds, [1, 2, 3, 4])
            self.assertIn(children[i].upperBounds, [10, 20, 30, 40])
            self.assertAlmostIn(children_sigmas[i], [0.1, 0.2, 0.3, 0.4])


    def test_recombine_default_strategy(self):
        """
        Tests the method ES_MuLambd.recombine().

        Checks if no recombination method is configured, then
        use a default strategy (just copy one of the parents).
        This test do not verify the integrity of the sigmas
        array returned.
        """
        TestsSupport.set_rule_discovery_configs(recombination=None)
        classifiers = TestsSupport.mock_classifiers(10)

        rule_disc = ES_MuLambd([])
        sigmas = rule_disc.extract_classifier_attributes(classifiers, x_dim=1, row=2)
        children, children_sigmas = rule_disc.recombine(classifiers, sigmas=sigmas)

        self.assertEqual(len(children_sigmas), len(children))
        self.assertEqual(children_sigmas.shape, (1,))


    # ------------- mutate_and_fit() --------------


    def test_mutate_and_fit(self):
        """
        Tests the method ES_MuLambd.mutate_and_fit().

        Checks if the lower and upper bounds were slightly
        ( <= (u - l) / 10 ) changed and if there is an
        error present.
        """
        n = 2
        optimizer = ES_MuLambd(pool=[])
        classifiers = TestsSupport.mock_specific_classifiers([ [[5], [10]], [[10], [5]] ])
        sigmas = optimizer.extract_classifier_attributes(classifiers, x_dim=1, row=2)
        X, y = TestsSupport.generate_input(n)
        mutated_children, mutated_sigmas = optimizer.mutate_and_fit(classifiers, X, y, sigmas)

        self.assertLessEqual(classifiers[0].lowerBounds, 5 + (10 - 5 / 10))


    # ------------- replace() --------------


    def test_replace_plus_classifiers(self):
        """
        Tests the method ES_MuLambd.replace().

        Chaecks that the + replacement operator
        is propperly returning both parents and
        children.
        """
        TestsSupport.set_rule_discovery_configs(replacement='+')
        optimizer = ES_MuLambd(pool=[])

        parents = TestsSupport.mock_classifiers(5)
        parents_sigmas = optimizer.extract_classifier_attributes(parents, x_dim=1, row=2)
        children = TestsSupport.mock_classifiers(3)
        children_sigmas = optimizer.extract_classifier_attributes(children, x_dim=1, row=2)

        array_concatenation = np.concatenate((children, parents))
        replacement_array, _ = optimizer.replace(parents, parents_sigmas, children, children_sigmas)
        self.assertTrue(len(array_concatenation) == len(replacement_array))
        for i in range(len(array_concatenation)):
            replaced_cl_bounds = (replacement_array[i].lowerBounds, replacement_array[i].upperBounds)
            concatenated_cl_bounds = (array_concatenation[i].lowerBounds, array_concatenation[i].upperBounds)
            self.assertTrue(np.allclose(replaced_cl_bounds, concatenated_cl_bounds))


    def test_replace_plus_sigmas(self):
        """
        Tests the method ES_MuLambd.replace().

        Chaecks that the + replacement operator
        is propperly returning the sigmas for both
        parents and children.
        """
        TestsSupport.set_rule_discovery_configs(replacement='+')
        optimizer = ES_MuLambd(pool=[])

        parents = TestsSupport.mock_classifiers(5)
        parents_sigmas = optimizer.extract_classifier_attributes(parents, x_dim=1, row=2)
        children = TestsSupport.mock_classifiers(3)
        children_sigmas = optimizer.extract_classifier_attributes(children, x_dim=1, row=2)

        array_concatenation = np.append(parents_sigmas, children_sigmas)
        _, replacement_array = optimizer.replace(parents, parents_sigmas, children, children_sigmas)

        self.assertTrue(np.allclose(replacement_array, array_concatenation))


if __name__ == '__main__':
    unittest.main()
