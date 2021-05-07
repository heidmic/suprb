import numpy as np

from hypothesis import given, settings
from hypothesis.strategies import lists, integers, floats
from suprb2 import Classifier, Config


@given(
    lists(floats(min_value=-1, max_value=1, allow_nan=False), min_size=1,
          max_size=10))  # points
@settings(max_examples=500)
def test_random_cl_with_point(point):
    """
    Place random classifiers around given points and check it they fall
    within bounds and have a volume
    :param point:
    :return:
    """
    Config().xdim = len(point)
    point = np.array(point, dtype=np.float64)
    cl = Classifier.random_cl(point)
    assert len(cl.upperBounds) == len(cl.lowerBounds) == len(point)
    assert (cl.upperBounds - cl.lowerBounds > 0).all()
    assert (cl.upperBounds <= 1).all()
    assert (cl.lowerBounds >= -1).all()
    assert (cl.lowerBounds <= point).all()
    assert (cl.upperBounds >= point).all()


@given(integers(min_value=1, max_value=10))
@settings(max_examples=500)
def test_random_cl(xdim):
    """
    Place random classifiers and check it they fall within bounds and have a
    volume
    :param point:
    :return:
    """
    Config().xdim = xdim
    cl = Classifier.random_cl()
    assert len(cl.upperBounds) == len(cl.lowerBounds) == xdim
    assert (cl.upperBounds - cl.lowerBounds > 0).all()
    assert (cl.upperBounds <= 1).all()
    assert (cl.lowerBounds >= -1).all()


