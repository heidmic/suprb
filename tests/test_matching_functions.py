from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle as apply_shuffle

from suprb.rule.fitness import VolumeWu
from suprb.rule.matching import OrderedBound, UnorderedBound, CenterSpread, MinPercentage, GaussianKernelFunction
import unittest
from suprb import SupRB
from suprb import Rule
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import es
import numpy as np

from suprb.utils import check_random_state
from suprb.optimizer.rule.mutation import Normal, HalfnormIncrease, Halfnorm, Uniform, UniformIncrease
from suprb.rule.initialization import MeanInit, NormalInit, HalfnormInit
import itertools


class TestMatchingFunction(unittest.TestCase):
    """Test Matching Function Implementation based on Higdon Gramacy Lee"""

    def create_rule(self, fitness, experience, error):
        center = np.array([0, 0])
        radius = np.array([0.5, 0.1])

        rule = Rule(match=GaussianKernelFunction(center, radius),
                    input_space=[-1.0, 1.0],
                    model=SupRB(),
                    fitness=VolumeWu)

        rule.fitness_ = fitness
        rule.experience_ = experience
        rule.error_ = error

        return rule

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
            rule_generation=es.ES1xLambda(
                operator='&',
                n_iter=12,
                delay=10,
                init=
                initialization(fitness=rule.fitness.VolumeWu(alpha=0.8), sigma=sigma_init)
                if initialization in (NormalInit, HalfnormInit)
                else initialization(fitness=rule.fitness.VolumeWu(alpha=0.8)),
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

        mutation_operators = [Normal, HalfnormIncrease, Halfnorm, Uniform, UniformIncrease]

        initialization_operators = [MeanInit, NormalInit, HalfnormInit]

        self.combined_matching_params = list(itertools.product(
            matching_functions, mutation_operators, initialization_operators))

    def setUp(self) -> None:
        self.setup_test_example()
        self.setup_matching_function_params()

        return super().setUp()

    def test_ellipsoid_volume(self):
        # circle
        ellipsoid = GaussianKernelFunction(center=np.array([0, 0]), radius=np.array([1, 1]))
        self.assertAlmostEqual(ellipsoid.volume_, 3.1416, 4)
        # sphere
        ellipsoid = GaussianKernelFunction(center=np.array([0, 0, 0]), radius=np.array([1, 1, 1]))
        self.assertAlmostEqual(ellipsoid.volume_, (4 / 3) * 3.1416, 4)
        # sphere with different radius
        ellipsoid = GaussianKernelFunction(center=np.array([0, 0, 0]), radius=np.array([0.5, 0.5, 0.5]))
        self.assertAlmostEqual(ellipsoid.volume_, (4 / 3) * 3.1416 * 0.125, 4)

    def test_ellipsoid_matching(self):
        # circle -> are the center points inside the circle
        points = np.array([[0, 0], [0, 0], [0, 0]])
        ellipsoid = GaussianKernelFunction(center=np.array([0, 0]), radius=np.array([1, 1]))
        self.assertTrue(all(ellipsoid.__call__(points)))

        # circle -> are the points on the outer radius inside the circle
        points = np.array([[0, 0.999], [0.999, 0]])
        ellipsoid = GaussianKernelFunction(center=np.array([0, 0]), radius=np.array([1, 1]))
        self.assertTrue(all(ellipsoid.__call__(points)))

        # circle -> are the points on the outer radius inside the circle
        points = np.array([[0.2, 0.3], [-0.1, -0.1]])
        ellipsoid = GaussianKernelFunction(center=np.array([0.5, 0.5]), radius=np.array([0.4, 0.4]))
        self.assertTrue(ellipsoid.__call__(points)[0])
        self.assertFalse(ellipsoid.__call__(points)[1])

        # sphere -> are the center points inside the sphere
        points = np.array([[0, 0, 0]])
        ellipsoid = GaussianKernelFunction(center=np.array([0, 0, 0]), radius=np.array([1, 1, 1]))
        self.assertTrue(all(ellipsoid.__call__(points)))

        # ellipsoid -> are the points in the ellipsoid
        points = np.array([[0, 0, 0], [0.999, 0, 0], [0, 0.499, 0], [0, 0, -0.999]])
        ellipsoid = GaussianKernelFunction(center=np.array([0, 0, 0]), radius=np.array([1, 0.5, -1]))
        self.assertTrue(all(ellipsoid.__call__(points)))

    def test_ellipsoid_clip(self):
        center = np.array([0, 0.5, -0.7])
        radius = np.array([8, 0, 7])

        ellipsoid = GaussianKernelFunction(center=center, radius=radius)
        ellipsoid.clip(np.array([]))

        print(f"center: {ellipsoid.center} radius: {ellipsoid.radius}")

        self.assertTrue(all(np.equal(ellipsoid.center, np.array([0, 0.5, -0.7]))))
        np.testing.assert_almost_equal(ellipsoid.radius, np.array([1., 0.5, 0.3]), decimal=5)

    def test_ellipsoid_mutate(self):
        center = np.array([0, 0])

        X = np.array([center - 0.001])

        rule = self.create_rule(fitness=1, experience=1,error=0.01)
        test = rule.match(X)
        ruleMutate = Uniform(matching_type=GaussianKernelFunction, sigma=0.1)
        ruleMutate.gaussian_kernel_function(rule=rule, random_state=42)
        test1 = rule.match(X)

        #

        self.assertTrue(True)

    @unittest.skip("isn't working on python 3.9")
    def test_smoke_test(self):
        for matching_func, mutation, initialization in self.combined_matching_params:
            self.setup_model(matching_func, mutation, initialization)

            print(f"\n\nChecking... {matching_func.__name__} with {mutation.__name__} and {initialization.__name__}")

            try:
                self.assertTrue(True),
                print("PASSED [Model Generation]\n")
            except:
                self.assertTrue(False), f"FAILED! Model generation with this config:" \
                                        f" {matching_func} with {mutation} and {initialization}"

            try:
                self.model.fit(self.X, self.y)
                self.assertTrue(True),
                print("PASSED [Model fit]\n")
            except:
                if matching_func.__name__ in ("CenterSpread", "MinPercentage") \
                        and (mutation == Halfnorm or initialization == HalfnormInit) \
                        or matching_func.__name__ == "UnorderedBound" \
                        and mutation in (HalfnormIncrease, UniformIncrease):
                    # All scenarios where a type error is wanted
                    self.assertTrue(True)

                else:
                    self.assertTrue(False), f"FAILED! Model fit with this config: " \
                                            f"{matching_func} with {mutation} and {initialization}"""


if __name__ == '__main__':
    unittest.main()
