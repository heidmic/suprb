from suprb2.config import Config
from suprb2.random_gen import Random
from suprb2.pool import ClassifierPool
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
        child_1 = Classifier(lowerBoundary=1, upperBoundary=1,
                                local_model= None, degree=1)
        """
        return [Classifier(i, i, None, 1) for i in range(1, n)]


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


    def initiate_pool(n: int, seed: int):
        """
        Initiate the ClassifierPool with n classifiers
        and return the auto generated samples used to
        fit the classifier. The classifier's error is
        either -1 or 0.

        X = Random().random.uniform(-2.5, 7, (n, 1))
        y = 0.75 * X**3 - 5 * X**2 + 4 * X + 12
        """
        Config().xdim = 1
        Random().reseed(seed)
        X, y = TestsSupport.generate_input(n)
        for x in X:
            cl = Classifier.random_cl(x)
            cl.fit(X, y)
            cl.error = np.random.randint(-1, 0)
            ClassifierPool().classifiers.append(cl)
        return (X, y)


    def set_rule_discovery_configs(**configs):
        """
        Sets the specified rule dicovery configurations.
        """
        for key, value in configs.items():
            Config().rule_discovery[key] = value
