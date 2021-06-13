from suprb2.config import Config
import numpy as np

from hypothesis import given, settings, strategies as st
from hypothesis.strategies import lists, integers, tuples, booleans, floats
from sklearn.linear_model import LinearRegression
from suprb2 import Individual, Classifier


def create_classifier(experience, error, lower, upper):
    Config().classifier['local_model'] = 'linear_regression'
    classifier = Classifier(lower, upper, 1)
    classifier.error = error
    classifier.experience = experience

    return classifier


def create_data(lower, upper, step):
    data = np.arange(lower, upper, step)
    return np.reshape(data, (len(data), 1))


input_data = create_data(-1, 1, 0.1)


@given(
    lists(
        st.tuples(
            integers(min_value=0),                     # experience
            floats(min_value=0, allow_nan=False))))  # error
@settings(max_examples=100)
def test_calculate_mixing_weights(input_parameter):
    '''
    Test calculate_mixing_weights function given
    - experience in range [0, inf]
    - error      in range [0, inf]
    '''

    # Given
    # We don't need the individuals genomes for this test, so we can leave the genome length as 0
    expected_result = np.zeros(len(input_parameter))
    classifier_pool = list()

    for i in range(len(input_parameter)):
        experience = input_parameter[i][0]
        error = input_parameter[i][1]

        classifier_pool.append(create_classifier(experience, error, -1, 1))

        if experience == 0:
            expected_result[i] = 0
        elif error == 0:
            expected_result[i] = np.inf
        else:
            expected_result[i] = experience/error

    individual = Individual.random_individual(10000, classifier_pool)

    # When
    result = individual.calculate_mixing_weights(classifier_pool)

    # Then
    assert len(result) == len(expected_result)
    assert np.isclose(result, expected_result).all()


@given(
    lists(
        st.tuples(
            floats(min_value=-1, max_value=1, allow_nan=False),   # lower bound
            floats(min_value=-1, max_value=1, allow_nan=False),   # upper bound
            booleans())))                                         # genome
@settings(max_examples=100)
def test_predict_classifier(input_parameter):
    """
    Tests predict for the function f(x) = x
    """

    classifier_pool = list()

    for i in range(len(input_parameter)):
        lower = input_parameter[i][0]
        upper = input_parameter[i][1]
        genome = input_parameter[i][2]

        classifier = create_classifier(1, 1e-4, lower, upper)
        classifier.fit(input_data, input_data)
        classifier_pool.append(classifier)

    individual = Individual.random_individual(len(classifier_pool), classifier_pool)

    for i in range(len(classifier_pool)):
        individual.genome[i] = genome

    # When
    result = individual.predict(input_data)

    bounds_check = False

    for i in range(len(input_parameter)):
        genome_check = (individual.genome[i] == True)
        lower_check = (input_data >= input_parameter[i][0])
        upper_check = (input_data <= input_parameter[i][1])

        bounds_check = bounds_check | (genome_check & lower_check & upper_check)

    expected_result = np.where(bounds_check, input_data, 0.0)

    # Then
    np.testing.assert_array_almost_equal(expected_result, result)
