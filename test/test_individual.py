from suprb2.config import Config
from suprb2.pool import ClassifierPool
from suprb2.individual import Individual
from suprb2.classifier import Classifier
from test.tests_support import TestsSupport

import unittest
import numpy as np

class TestIndividual(unittest.TestCase):
    """
    This module test all methods from Individual
    """


    def setUp(self):
        """
        Resets the hyper parameters for the next test.
        """
        Config().__init__()


    # ------------- Individual.get_classifiers() --------------


    def test_get_classifiers_unmatched(self):
        """
        Tests the method Individual.get_classifiers().

        When get_classifiers(unmatched=True),
        then this should return the classifiers that
        were not considered in the last solution.
        """
        # Start the ClassifierPool
        n = 10
        ClassifierPool().classifiers = [Classifier.random_cl(None, 1) for i in range(n)]

        # Get classifiers
        ind = Individual.random_individual(n)
        classifiers_in_individual = ind.get_classifiers()
        classifiers_out_individual = ind.get_classifiers(unmatched=True)

        # Compare classifiers
        for cl in classifiers_out_individual:
            self.assertNotIn(cl, classifiers_in_individual)
