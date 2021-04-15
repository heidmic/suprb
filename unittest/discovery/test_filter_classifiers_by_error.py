from suprb2.config import Config
from suprb2.random_gen import Random
from suprb2.pool import ClassifierPool
from suprb2.classifier import Classifier
from suprb2.discovery import RuleDiscoverer

import unittest
import numpy as np

class TestFilterClassifiersByError(unittest.TestCase):
    def test_if_error_is_take_into_account(self):
        """
        Test that classifiers with an error bigger
        than the default error
        """
        Config().xdim = 1

        n = 10
        X = Random().random.uniform(-2.5, 7, (n, 1))
        y = 0.75*X**3-5*X**2+4*X+12

        for x in X:
            cl = Classifier.random_cl(x)
            cl.fit(X, y)
            cl.error = np.random.randint(-1, 0)
            ClassifierPool().classifiers.append(cl)

        optimizer = RuleDiscoverer()
        filtered_cls = optimizer.filter_classifiers_by_error(ClassifierPool().classifiers, 10, X, y)

        for cl in filtered_cls:
            if cl.error < optimizer.default_error(y):
                self.assertIn(cl, filtered_cls)
            else:
                self.assertNotIn(cl, filtered_cls)
        ClassifierPool().classifiers = list()




if __name__ == '__main__':
    unittest.main()
