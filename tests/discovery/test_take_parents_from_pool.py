from tests.examples import Examples
from suprb2.pool import ClassifierPool
from suprb2.discovery import RuleDiscoverer

import unittest

class TestTakeParentsFromPool(unittest.TestCase):
    """
    This module focuses on testing the method
    RuleDiscoverer.take_parents_from_pool
    """

    def test_parents_are_removed_from_pool(self):
        """
        Test that parents are no longer in the pool
        after the operation is over.
        """
        mu = 10
        X, y = Examples.initiate_pool(mu)
        Examples.set_rule_discovery_configs(mu=mu)

        optimizer = RuleDiscoverer()
        parents = optimizer.take_parents_from_pool()

        self.assertEqual(ClassifierPool().classifiers, [])


if __name__ == '__main__':
    unittest.main()
