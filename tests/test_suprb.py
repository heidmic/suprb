import unittest
import numpy as np

from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils import shuffle as apply_shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import suprb
import suprb.logging.stdout
from suprb.rule.matching import OrderedBound
from suprb.utils import check_random_state

from suprb import SupRB
from suprb import rule
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import es
from suprb.utils import check_random_state
import suprb.optimizer.rule.mutation as mutation


class TestSupRB(unittest.TestCase):
    def setUp(self):
        self.model_ = SupRB(
            rule_generation=es.ES1xLambda(
                n_iter=220,
                operator='&',
                init=rule.initialization.MeanInit(fitness=rule.fitness.VolumeWu(alpha=0.05)),
                mutation=mutation.HalfnormIncrease(sigma=0.01, adapt_mutation=True),
                delay=200
            ),
            solution_composition=ga.GeneticAlgorithm(
                n_iter=1,
                crossover=ga.crossover.Uniform(),
                selection=ga.selection.Tournament(),
            ),
            n_iter=1,
            n_rules=5,
            verbose=1,
            random_state=42,
        )

        self.setup_test_example()

    def setup_test_example(self):
        n_samples = 1000
        random_state = 42
        random_state_ = check_random_state(random_state)

        X = np.linspace(0, 20, num=n_samples)
        y = np.zeros(n_samples)
        y[X < 10] = np.sin(np.pi * X[X < 10] / 5) + 0.2 * np.cos(4 * np.pi * X[X < 10] / 5)
        y[X >= 10] = X[X >= 10] / 10 - 1
        y += random_state_.normal(scale=0.1, size=n_samples)
        X = X.reshape((-1, 1))
        X, y = apply_shuffle(X, y, random_state=random_state)

        self.X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
        self.y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    # def test_check_estimator(self):
    #     """Tests that `check_estimator()` from sklearn passes,
    #      i.e., that the scikit-learn interface guidelines are met."""

    #     check_estimator(self.model_)

    def test_application(self):
        """Tests if suprb is learning continously or if something 
        breaks in between (e.g. it stops learning)."""
        self.model_.n_iter = 10
        self.model_.fit(self.X, self.y)

        # self.model_._discover_rules(self.X, self.y, self.model_.n_rules)
        # self.model_._compose_solution(self.X, self.y)

        # # for i in self.model_.pool_:
        # #     print(i.error_, i.experience_, i.fitness_)

        # self.model_._discover_rules(self.X, self.y, self.model_.n_rules)
        # self.model_._compose_solution(self.X, self.y)

        # for i in self.model_.pool_:
        #     print(i.error_, i.experience_, i.fitness_)
