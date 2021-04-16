from suprb2.config import Config
from suprb2.random_gen import Random
from suprb2.pool import ClassifierPool
from suprb2.classifier import Classifier
from suprb2.discovery import RuleDiscoverer

import numpy as np

class Examples:
    """
    This module serves as a container with helpful methods
    for generating the necessary data for tests.
    """

    def mock_classifiers(n):
        """
        Creates n classifiers without relevant attributes.
        Example:
        child_1 = Classifier(lowerBoundary=1, upperBoundary=1,
                                local_model= None, degree=1)
        """
        return [Classifier(i, i, None, 1) for i in range(1, n)]


    def initiate_pool(n):
        """
        Initiate the ClassifierPool with n classifiers
        and return the auto generated samples used to
        fit the classifier. The classifier's error is
        either -1 or 0.

        X = Random().random.uniform(-2.5, 7, (n, 1))
        y = 0.75 * X**3 - 5 * X**2 + 4 * X + 12
        """
        Config().xdim = 1
        X = Random().random.uniform(-2.5, 7, (n, 1))
        y = 0.75*X**3-5*X**2+4*X+12

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


    def reset_enviroment():
        """
        Resets the Classifier Pool for the next test.
        """
        ClassifierPool().classifiers = list()
