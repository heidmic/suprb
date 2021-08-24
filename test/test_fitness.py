import unittest


from suprb2.classifier import Classifier
from suprb2.config import Config
from suprb2 import LCS
from test.tests_support import TestsSupport

fitness_functions = ["pseudo-BIC",
                     "BIC_matching_punishment",
                     "mse",
                     "mse_times_C",
                     "mse_times_root_C",
                     "mse_matching_pun",
                     # TODO: f1_score does not work with regression -> How to test that this works?
                     #  "inverted_macro_f1_score",
                     #  "inverted_macro_f1_score_times_C"
                     ]


def create_classifier(experience, error, lower, upper):
    Config().classifier['local_model'] = 'linear_regression'
    classifier = Classifier(lower, upper, Config())
    classifier.error = error
    classifier.experience = experience

    return classifier


def test_pseudo_BIC():
    """
    Tests if the pseudo-BIC fitness function compiles
    """

    Config().rule_discovery["recombination"] = "u"
    X, y = TestsSupport.generate_input(10)

    for f in fitness_functions:
        print(f)
        Config().solution_creation["fitness"] = f

        lcs = LCS(1, Config(), logging=False)
        lcs.run_inital_step(X, y)

        assert(lcs.fitness_function, f)


if __name__ == '__main__':
    unittest.main()
