from tests.examples import Examples
from suprb2.pool import ClassifierPool
from suprb2.discovery import RuleDiscoverer

import unittest
import numpy as np

class TestFilterClassifiers(unittest.TestCase):
    """
    This module focuses on testing the method
    RuleDiscoverer.filter_classifier
    """

    def test_if_error_is_take_into_account(self):
        """
        Test if the errors of the classifiers, that were filtered,
        are greater than the default error.
        """
        X, y = Examples.initiate_pool(10)

        optimizer = RuleDiscoverer()
        filtered_cls = optimizer.filter_classifiers(ClassifierPool().classifiers, 10, X, y)
        for cl in ClassifierPool().classifiers:
            if cl.error < optimizer.default_error(y):
                self.assertIn(cl, filtered_cls)
            else:
                self.assertNotIn(cl, filtered_cls)

        Examples.reset_enviroment()


if __name__ == '__main__':
    unittest.main()
