import numpy as np

from hypothesis import given, settings, strategies as st
from hypothesis.strategies import lists, integers, decimals, tuples, booleans
from sklearn.linear_model import LinearRegression
from suprb2 import Individual, Classifier, ClassifierPool


def create_classifier(experience, error, lower, upper):
    classifier = Classifier(lower, upper, LinearRegression(), 1)
    classifier.error = error
    classifier.experience = experience

    return classifier


def create_data(lower, upper, step):
    data = np.arange(lower, upper, step)
    return np.reshape(data, (len(data), 1))


train_X = create_data(-1, 1, 0.01)
train_y = train_X

test_X = create_data(-1, 1, 0.1)
test_y = test_X


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

        classifier_list.append(create_classifier(experience, error, -1, 1))

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


@given(
    lists(
        st.tuples(
            decimals(min_value=-1, max_value=1, allow_nan=False),   # lower bound
            decimals(min_value=-1, max_value=1, allow_nan=False),   # upper bound
            booleans())))                                           # genome
@settings(max_examples=100)
def test_predict_classifier(input_parameter):
    """
    Tests predict for the function f(x) = x
    """

    ClassifierPool().classifiers = list()
    individual = Individual.random_individual(1000000)

    for i in range(len(input_parameter)):
        lower = float(input_parameter[i][0])
        upper = float(input_parameter[i][1])
        genome = input_parameter[i][2]

        classifier = create_classifier(1, 1e-4, lower, upper)
        classifier.fit(train_X, train_y)
        ClassifierPool().classifiers.append(classifier)

        individual.genome[i] = genome

    # When
    result = individual.predict(test_X)

    bounds_check = False

    for i in range(len(input_parameter)):
        bounds_check = bounds_check | ((individual.genome[i] == True) & (test_y >= float(input_parameter[i][0])) & (
            test_y <= float(input_parameter[i][1])))

    expected_result = np.where(bounds_check, test_y, 0.0)

    # Then
    np.testing.assert_array_almost_equal(expected_result, result)
