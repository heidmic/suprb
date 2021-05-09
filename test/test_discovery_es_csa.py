from suprb2.config import Config
from suprb2.discovery import ES_CSA
from suprb2.utilities import Utilities
from suprb2.classifier import Classifier
from test.tests_support import TestsSupport

import unittest
import numpy as np
from mock import patch, Mock

class TestDiscoveryES_CSA(unittest.TestCase):
    """
    This module test all methods from ES_CSA
    """

    # ------------- step() --------------


    def test_step(self):
        X, y = TestsSupport.generate_input(10)
        optimizer = ES_CSA(pool=[])
        cl, cl_sigmas = optimizer.step(X, y)
        self.assertIsInstance(cl, Classifier)
        self.assertEqual(cl_sigmas.shape, (1,))


    # ------------- select_best_classifiers() --------------


    def test_select_best_classifiers(self):
        n, mu = (10, 5)
        optimizer = ES_CSA(pool=[])
        X, y = TestsSupport.generate_input(n)
        classifiers, cls_sigmas = list(), np.zeros((10, X.shape[1]))
        for i in range(n):
            cl = Classifier.random_cl(X[i], xdim=X.shape[1])
            cl.error = i
            classifiers.append(cl)
            cls_sigmas[i] = optimizer.create_sigmas(X.shape[1])

        best_cls, best_sigmas = optimizer.select_best_classifiers(classifiers, cls_sigmas, mu)

        self.assertEqual(len(best_cls), len(best_sigmas))
        self.assertEqual(len(best_cls), mu)
        for i in range(mu):
            self.assertIn(classifiers[i], best_cls)



if __name__ == '__main__':
    unittest.main()
