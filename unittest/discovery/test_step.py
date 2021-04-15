from suprb2.config import Config
from suprb2.random_gen import Random
from suprb2.pool import ClassifierPool
from suprb2.classifier import Classifier
from suprb2.discovery import RuleDiscoverer

import unittest

class TestStep(unittest.TestCase):
    def test_mu_equals_lambda(self):
        """
        Test that when mu is equal to lambda, then the population
        will not grow nor sink.
        Classifier will only be added if their error is acceptable,
        that is why all errors are mocked with -1.
        """
        Config().xdim = 1
        Config().rule_discovery['mu'] = 10
        Config().rule_discovery['lmbd'] = 10
        Config().rule_discovery['selection'] = '+'
        Config().rule_discovery['steps_per_step'] = 1

        n = 10
        X = Random().random.uniform(-2.5, 7, (n, 1))
        y = 0.75*X**3-5*X**2+4*X+12

        for x in X:
            cl = Classifier.random_cl(x)
            cl.fit(X, y)
            cl.error = -1
            ClassifierPool().classifiers.append(cl)

        optimizer = RuleDiscoverer()
        optimizer.step(X, y)
        self.assertEqual(len(ClassifierPool().classifiers), 10)
        ClassifierPool().classifiers = list()


if __name__ == '__main__':
    unittest.main()
