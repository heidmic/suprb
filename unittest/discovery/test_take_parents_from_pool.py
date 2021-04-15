from suprb2.config import Config
from suprb2.random_gen import Random
from suprb2.pool import ClassifierPool
from suprb2.classifier import Classifier
from suprb2.discovery import RuleDiscoverer

import unittest

class TestTakeParentsFromPool(unittest.TestCase):
    def test_parents_are_removed_from_pool(self):
        """
        Test that parents are no longer in the pool
        after the operation is over.
        """
        Config().xdim = 1

        n = 10
        X = Random().random.uniform(-2.5, 7, (n, 1))
        y = 0.75*X**3-5*X**2+4*X+12

        for x in X:
            cl = Classifier.random_cl(x)
            cl.fit(X, y)
            ClassifierPool().classifiers.append(cl)

        optimizer = RuleDiscoverer()
        parents = optimizer.take_parents_from_pool()

        for p in parents:
            self.assertNotIn(p, ClassifierPool().classifiers)
        self.assertEqual(ClassifierPool().classifiers, [])



if __name__ == '__main__':
    unittest.main()
