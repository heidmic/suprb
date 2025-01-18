from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle as apply_shuffle

from suprb.rule.matching import (
    OrderedBound,
    UnorderedBound,
    CenterSpread,
    MinPercentage,
)
import unittest
from suprb import SupRB
from suprb import rule
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import es
import numpy as np

from suprb.utils import check_random_state
from suprb.optimizer.rule.mutation import (
    Normal,
    HalfnormIncrease,
    Halfnorm,
    Uniform,
    UniformIncrease,
)
from suprb.rule.initialization import MeanInit, NormalInit, HalfnormInit
import itertools


class TestMatchingFunction(unittest.TestCase):
    """Test Matching Function Implementation based on Higdon Gramacy Lee"""

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

    def setup_model(self, matching_func, mutation, initialization):
        sigma_mutate = 2
        sigma_init = 0.1
        if matching_func in (CenterSpread, MinPercentage):
            sigma_mutate = np.array([2, 2])
            sigma_init = np.array([0.1, 0.1])

        self.model = SupRB(
            rule_discovery=es.ES1xLambda(
                operator="&",
                n_iter=12,
                delay=10,
                init=(
                    initialization(fitness=rule.fitness.VolumeWu(alpha=0.8), sigma=sigma_init)
                    if initialization in (NormalInit, HalfnormInit)
                    else initialization(fitness=rule.fitness.VolumeWu(alpha=0.8))
                ),
                mutation=mutation(sigma=sigma_mutate),
            ),
            matching_type=matching_func(np.array([])),
            solution_composition=ga.GeneticAlgorithm(
                n_iter=16,
                crossover=ga.crossover.Uniform(),
                selection=ga.selection.Tournament(),
                mutation=ga.mutation.BitFlips(),
            ),
            n_iter=32,
            n_rules=4,
            logger=StdoutLogger(),
        )

    def setup_matching_function_params(self):

        matching_functions = [OrderedBound, UnorderedBound, CenterSpread, MinPercentage]

        mutation_operators = [
            Normal,
            HalfnormIncrease,
            Halfnorm,
            Uniform,
            UniformIncrease,
        ]

        initialization_operators = [MeanInit, NormalInit, HalfnormInit]

        self.combined_matching_params = list(
            itertools.product(matching_functions, mutation_operators, initialization_operators)
        )

    def setUp(self) -> None:
        self.setup_test_example()
        self.setup_matching_function_params()

        return super().setUp()

    def test_smoke_test(self):
        for matching_func, mutation, initialization in self.combined_matching_params:
            self.setup_model(matching_func, mutation, initialization)

            print(f"\n\nChecking... {matching_func.__name__} with {mutation.__name__} and {initialization.__name__}")

            try:
                self.assertTrue(True),
                print("PASSED [Model Generation]\n")
            except:
                self.assertTrue(
                    False
                ), f"FAILED! Model generation with this config:" f" {matching_func} with {mutation} and {initialization}"

            try:
                self.model.fit(self.X, self.y)
                self.assertTrue(True),
                print("PASSED [Model fit]\n")
            except:
                if (
                    matching_func.__name__ in ("CenterSpread", "MinPercentage")
                    and (mutation == Halfnorm or initialization == HalfnormInit)
                    or matching_func.__name__ == "UnorderedBound"
                    and mutation in (HalfnormIncrease, UniformIncrease)
                ):
                    # All scenarios where a type error is wanted
                    self.assertTrue(True)

                else:
                    self.assertTrue(
                        False
                    ), f"FAILED! Model fit with this config: " f"{matching_func} with {mutation} and {initialization}"


if __name__ == "__main__":
    unittest.main()
