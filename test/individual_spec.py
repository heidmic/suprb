from hypothesis import given, settings, strategies as st
from suprb2 import Individual, Classifier, ClassifierPool
import numpy as np
from sklearn.linear_model import LinearRegression
from hypothesis.strategies import lists, integers, decimals


def add_classifiers_to_pool(X, y, lower, upper):
    cl = Classifier(lower, upper, LinearRegression(), 1)
    cl.fit(X, y)
    ClassifierPool().classifiers.append(cl)


def get_sample_individual():
    classifier_count = len(ClassifierPool().classifiers)
    individual = Individual.random_individual(classifier_count)

    for i in range(classifier_count):
        individual.genome[i] = True

    return individual


def create_data(lower, upper, step):
    data = np.arange(lower, upper, step)
    return np.reshape(data, (len(data), 1))


@given(lists(lists(decimals(min_value=-1, max_value=1),  min_size=2, max_size=2), min_size=2, max_size=2))
@settings(max_examples=100)
def test_simple(classifiers):

    # Given
    ClassifierPool().classifiers = list()
    X = create_data(-0.9, 1.0, 0.01)
    y = X

    for classifier in classifiers:
        classifier = sorted(classifier, key=float)
        lower_bound = classifier[0]
        upper_bound = classifier[1]

        add_classifiers_to_pool(X, y, lower_bound, upper_bound)

    individual = get_sample_individual()

    # When
    result = individual.calculate_mixing_weights()

    # Then
    # TODO Not sure how to test this
    print(result)
