from tests.examples import Examples
from suprb2.pool import ClassifierPool
from suprb2.discovery import RuleDiscoverer

import unittest

class TestStep(unittest.TestCase):
    """
    This module focuses on testing the method
    RuleDiscoverer.step
    """

    def test_mu_equals_lambda(self):
        """
        Test that when mu is equal to lambda, then the population
        will not grow nor sink.
        Classifier will only be added if their error is acceptable,
        that is why all errors are mocked with -1.
        """
        Examples.set_rule_discovery_configs(mu=10, lmbd=10, selection='+', steps_per_step=1)
        X, y = Examples.initiate_pool(10)

        optimizer = RuleDiscoverer()
        optimizer.step(X, y)
        self.assertEqual(len(ClassifierPool().classifiers), 10)

        Examples.reset_enviroment()


if __name__ == '__main__':
    unittest.main()
