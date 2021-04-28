from suprb2.config import Config
from suprb2.classifier import Classifier
from tests.tests_support import TestsSupport

import unittest
import numpy as np

class TestClassifier(unittest.TestCase):
    """
    This module test all methods from Classifier
    """


    def setUp(self):
        """
        Resets the hyper parameters for the next test.
        """
        Config().__init__()


    # ------------- random_cl() --------------


    def test_mutation_vector_is_numpy(self):
        """
        Tests the method Classifier.random_cl().

        The mutation vector should be initialized
        with type np.ndarray
        """
        X, y = TestsSupport.generate_input(10)
        TestsSupport.set_rule_discovery_configs(sigma='vector')
        cl = Classifier.random_cl(X[0])
        self.assertEqual(type(cl.sigmas), np.ndarray)


    def test_mutation_vector_dtype(self):
        """
        Tests the method Classifier.random_cl().

        The mutation vector type should be np.float64
        """
        X, y = TestsSupport.generate_input(10)
        TestsSupport.set_rule_discovery_configs(sigma='vector')
        cl = Classifier.random_cl(X[0])
        self.assertEqual(cl.sigmas.dtype, np.float64)


    def test_mutation_vector_values(self):
        """
        Tests the method Classifier.random_cl().

        The mutation vector should be initialized
        with values in [-1, 1].
        """
        X, y = TestsSupport.generate_input(10)
        TestsSupport.set_rule_discovery_configs(sigma='vector')
        cl = Classifier.random_cl(X[0])
        for i_dim in range(Config().xdim):
            self.assertAlmostEqual(cl.sigmas[i_dim], 0, delta=1)


    # ------------- mutate() --------------


    def test_mutate_uses_vector(self):
        pass


if __name__ == '__main__':
    unittest.main()
