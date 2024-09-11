import sklearn
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_validate, train_test_split


from suprb.utils import check_random_state
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.solution.ga import GeneticAlgorithm
from suprb.wrapper import SupRBWrapper


def load_higdon_gramacy_lee(n_samples=1000, noise=0, random_state=None):
    random_state_ = check_random_state(random_state)

    X = np.linspace(0, 20, num=n_samples)
    y = np.zeros(n_samples)

    y[X < 10] = np.sin(np.pi * X[X < 10] / 5) + 0.2 * np.cos(4 * np.pi * X[X < 10] / 5)
    y[X >= 10] = X[X >= 10] / 10 - 1

    y += random_state_.normal(scale=noise, size=n_samples)
    X = X.reshape((-1, 1))

    return sklearn.utils.shuffle(X, y, random_state=random_state)


if __name__ == '__main__':
    random_state = 42

    X, y = load_higdon_gramacy_lee(noise=0.1, random_state=random_state)

    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    # Comparable with examples/example_2.py
    model = SupRBWrapper(print_config=True,

                         ## RULE GENERATION ##
                         rule_generation=ES1xLambda(),
                         rule_generation__n_iter=10,
                         rule_generation__lmbda=16,
                         rule_generation__operator='+',
                         rule_generation__delay=150,
                         rule_generation__random_state=random_state,
                         rule_generation__n_jobs=1,

                         ## SOLUTION COMPOSITION ##
                         solution_composition=GeneticAlgorithm(),
                         solution_composition__n_iter=32,
                         solution_composition__population_size=32,
                         solution_composition__elitist_ratio=0.2,
                         solution_composition__random_state=random_state,
                         solution_composition__n_jobs=1)

    scores = cross_validate(model, X, y, cv=4, n_jobs=1, verbose=10,
                            scoring=['r2', 'neg_mean_squared_error'],
                            return_estimator=True, fit_params={'cleanup': True})
