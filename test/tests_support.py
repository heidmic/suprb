from suprb2.config import Config
from suprb2.random_gen import Random
from suprb2.classifier import Classifier
from suprb2.discovery import RuleDiscoverer

import numpy as np

class TestsSupport:
    """
    This module serves as a container with helpful methods
    for generating the necessary data for tests.
    """

    def mock_classifiers(n: int):
        """
        Creates n classifiers without relevant attributes.
        Example:
        child_1 = Classifier(lowerBoundary=[1], upperBoundary=[1],
                                local_model= None, degree=1)
        """
        return [ Classifier(np.array([1], dtype=np.float64),
                            np.array([1], dtype=np.float64),
                            None, 1) for i in range(1, n) ]


    def mock_specific_classifiers(values: list):
        """
        'value' is an array, where each line is represents:
        [ classifier.lowerBounds, classifier.upperBounds ]

        Example:
        value = [
            [1, 1],  # Classifier(lowerBounds=1, upperBounds=1, local_model=None, degree=1)
            [2, 1]   # Classifier(lowerBounds=2, upperBounds=1, local_model=None, degree=1)
        ]
        """
        classifiers = []
        for i in range(len(values)):
            classifiers.append(Classifier(values[i][0], values[i][1], None, 1))
        return classifiers


    def generate_input(n):
        X = np.random.uniform(-2.5, 7, (n, 1))
        y = 0.75*X**3-5*X**2+4*X+12
        return (X, y)


    def set_rule_discovery_configs(**configs):
        """
        Sets the specified rule dicovery configurations.
        """
        for key, value in configs.items():
            Config().rule_discovery[key] = value
