from suprb2.discovery import ES_MuLambdSearchPath
from suprb2.classifier import Classifier
from test.tests_support import TestsSupport

import unittest
import numpy as np
from mock import patch, Mock

class TestDiscoveryES_MuLambdSearchPath(unittest.TestCase):
    """
    This module test all methods from ES_MuLambdSearchPath
    """

    # ------------- step() --------------


    def test_step(self):
        TestsSupport.set_rule_discovery_configs(steps_per_step=5, mu=5, lmbd=10)
        X, y = TestsSupport.generate_input(10)
        optimizer = ES_MuLambdSearchPath(pool=[])
        optimizer.step(X, y)
        self.assertEqual(len(optimizer.pool), 1)


    # ------------- select_best_classifiers() --------------


    def test_select_best_classifiers(self):
        n, mu = (10, 5)
        optimizer = ES_MuLambdSearchPath(pool=[])
        X, y = TestsSupport.generate_input(n)
        pool = list()
        for i in range(n):
            cl = Classifier.random_cl(point=X[i], xdim=X.shape[1])
            cl.error = i
            pool.append([cl, optimizer.create_sigmas(X.shape[1])])

        best_tuples = optimizer.select_best_classifiers(pool, mu)
        self.assertEqual(len(best_tuples), mu)
        self.assertEqual(len(best_tuples[0]), 2)



if __name__ == '__main__':
    unittest.main()
