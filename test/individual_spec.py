import numpy as np

from hypothesis import given, settings, strategies as st
from hypothesis.strategies import lists, integers, decimals, tuples
from sklearn.linear_model import LinearRegression
from suprb2 import Individual, Classifier


def create_classifier(experience, error):
    classifier = Classifier(-1, 1, LinearRegression(), 1)
    classifier.error = error
    classifier.experience = experience

    return classifier


@given(
    lists(
        st.tuples(
            integers(min_value=0),                     # experience
            decimals(min_value=0, allow_nan=False))))  # error
@settings(max_examples=100)
def test_calculate_mixing_weights(input_parameter):
    '''
    Test calculate_mixing_weights function given
    - experience in range [0, inf]
    - error      in range [0, inf]
    '''

    # Given
    # We don't need the individuals genomes for this test, so we can leave the genome length as 0
    individual = Individual.random_individual(0)
    expected_result = np.zeros(len(input_parameter))
    classifier_list = list()

    for i in range(len(input_parameter)):
        experience = float(input_parameter[i][0])
        error = float(input_parameter[i][1])

        classifier_list.append(create_classifier(experience, error))

        if experience == 0:
            expected_result[i] = 0
        elif error == 0:
            expected_result[i] = np.inf
        else:
            expected_result[i] = experience/error

    # When
    result = individual.calculate_mixing_weights(classifier_list)

    # Then
    assert len(result) == len(expected_result)
    assert np.isclose(result, expected_result).all()
