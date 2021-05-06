import numpy as np
import unittest
import os

from hypothesis import given, settings, strategies as st
from hypothesis.strategies import lists, integers, decimals, tuples, booleans
from sklearn.linear_model import LinearRegression
from suprb2 import LCS, Classifier, ClassifierPool, Config, Individual


def create_classifier(experience, error):
    classifier = Classifier(-1, 1, LinearRegression(), 1)
    classifier.error = error
    classifier.experience = experience

    return classifier


def create_individual(individual_values):
    individual = Individual.random_individual(100000)
    individual.fitness = float(individual_values[0])
    individual.error = float(individual_values[1])
    individual.genome = individual_values[2]

    return individual


@given(
    lists(
        st.tuples(
            integers(min_value=0),                     # experience
            decimals(min_value=0, allow_nan=False)),   # error
        min_size=1),
    st.tuples(decimals(min_value=0, allow_nan=False),  # expected_elitist_values fitness
              decimals(min_value=0, allow_nan=False),  # expected_elitist_values error
              booleans()),                             # expected_elitist_values genome
    st.tuples(decimals(min_value=0, allow_nan=False),  # different_elitist_values fitness
              decimals(min_value=0, allow_nan=False),  # different_elitist_values error
              booleans()))                             # different_elitist_values genome
@settings(max_examples=100)
def test_save_and_load_model(classifier_values, expected_elitist_values, different_elitist_values):
    '''
    Test saving and loading of the model for
    - All Classifiers
    - Elitist
    '''

    # Given
    Config().save_model = True
    Config().load_model = False
    Config().model_name = "test_model.joblib"

    for i in range(len(classifier_values)):
        experience = float(classifier_values[i][0])
        error = float(classifier_values[i][1])

        ClassifierPool().classifiers.append(create_classifier(experience, error))

    expected_classifiers = ClassifierPool().classifiers
    expected_elitist = create_individual(expected_elitist_values)
    different_elitist = create_individual(different_elitist_values)

    lcs = LCS(1)
    lcs.set_elitist(expected_elitist)

    # When
    lcs.save_model()
    lcs.set_elitist(different_elitist)
    Config().load_model = True
    lcs.load_model()

    # Then
    assert len(ClassifierPool().classifiers) == len(expected_classifiers)

    assert expected_elitist.fitness == lcs.get_elitist().fitness
    assert expected_elitist.error == lcs.get_elitist().error
    assert expected_elitist.genome == lcs.get_elitist().genome

    for i in range(len(classifier_values)):
        assert ClassifierPool().classifiers[i].error == expected_classifiers[i].error
        assert ClassifierPool().classifiers[i].experience == expected_classifiers[i].experience

    os.remove(Config().model_name)
