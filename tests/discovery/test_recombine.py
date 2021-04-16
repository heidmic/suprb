from tests.examples import Examples
from suprb2.classifier import Classifier
from suprb2.discovery import RuleDiscoverer

import unittest

class TestRecombine(unittest.TestCase):
    """
    This module focuses on testing the method
    RuleDiscoverer.recombine
    """

    def test_average(self):
        """
        Tests if the average of the classifiers' boundaries
        is propperly calculated.

        child.lowerBound = average(parents.lowerBounds)
        child.upperBound = average(parents.upperBounds)
        """
        Examples.set_rule_discovery_configs(recombination='intermediate')
        child = RuleDiscoverer().recombine(Examples.mock_classifiers(10))
        self.assertEquals((child.lowerBounds, child.upperBounds), (5, 5))


    def test_default_strategy(self):
        """
        Tests if no recombination method is configured, then
        use a default strategy (just copy one of the parents).
        This test do not verify the integrity of the deepcopy,
        instead it just checks that the generated child is a
        Classifier.
        """
        parents = Examples.mock_classifiers(10)
        child = RuleDiscoverer().recombine(parents)
        self.assertIsInstance(child, Classifier)


if __name__ == '__main__':
    unittest.main()
