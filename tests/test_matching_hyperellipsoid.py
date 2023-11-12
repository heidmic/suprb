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
    """Test components of the Hyperellipsoidal Matching Function"""
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

    def setUp(self) -> None:
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

        self.assertTrue(all(np.equal(ellipsoid.center, np.array([0, 0.5, -0.7]))))
        np.testing.assert_almost_equal(ellipsoid.radius, np.array([1., 0.5, 0.3]), decimal=5)

    def test_ellipsoid_min_range(self):
        center = np.array([0, 0, 0])
        radius = np.array([0, 0, 0])

        ellipsoid = GaussianKernelFunction(center=center, radius=radius)
        ellipsoid.min_range(0.5)

        self.assertTrue(all(np.equal(ellipsoid.center, np.array([0, 0, 0]))))
        np.testing.assert_almost_equal(ellipsoid.radius, np.array([0.25, 0.25, 0.25]), decimal=5)


    def test_ellipsoid_mutate(self):
        center = np.array([0, 0])
        X = np.array([center - 0.001])

        rule = self.create_rule(fitness=1, experience=1,error=0.01)
        ruleMutate = Uniform(matching_type=GaussianKernelFunction, sigma=0.1)
        ruleMutate.gaussian_kernel_function(rule=rule, random_state=42)
        matched = rule.match(X)


if __name__ == '__main__':
    unittest.main()
