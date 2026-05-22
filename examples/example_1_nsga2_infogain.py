import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_validate, train_test_split

from suprb import SupRB
from suprb.utils import check_random_state
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.solution.ga import GeneticAlgorithm

from suprb.optimizer.rule.nsga2 import NSGA2InfoGain
from utils import log_scores

import numpy as np

from sklearn.linear_model import Ridge
from sklearn.utils import Bunch, shuffle

from suprb import rule, SupRB
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import origin, mutation

from suprb.optimizer.rule.ns.novelty_calculation import NoveltyCalculation
from suprb.optimizer.rule.ns.novelty_search_type import MinimalCriteria




def load_higdon_gramacy_lee(n_samples=1000, noise=0, random_state=None):
    random_state_ = check_random_state(random_state)

    X = np.linspace(0, 20, num=n_samples)
    y = np.zeros(n_samples)

    y[X < 10] = np.sin(np.pi * X[X < 10] / 5) + 0.2 * np.cos(4 * np.pi * X[X < 10] / 5)
    y[X >= 10] = X[X >= 10] / 10 - 1

    y += random_state_.normal(scale=noise, size=n_samples)
    X = X.reshape((-1, 1))

    return sklearn.utils.shuffle(X, y, random_state=random_state)


def create_plot(scores):
    fig, axes = plt.subplots(2, 2)
    X_plot = np.linspace(X.min(), X.max(), 500).reshape((-1, 1))
    for ax, model in zip(axes.flatten(), scores["estimator"]):
        pred = model.predict(X_plot)
        ax.scatter(X, y, c="b", s=3, label="y_true")
        ax.plot(X_plot, pred, c="r", label="y_pred")

    plt.savefig("result.png")


if __name__ == "__main__":
    random_state = 42

    X, y = load_higdon_gramacy_lee(noise=0.1, random_state=random_state)

    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    model = SupRB(
        rule_discovery=NSGA2InfoGain(
            n_iter=16,
            mu=16,
            lmbda=64,
            origin_generation=origin.SquaredError(),
            init=rule.initialization.MeanInit(
                fitness=rule.fitness.MooFitness(),
                model=Ridge(alpha=0.01, random_state=random_state),
                # matching_type=rule.matching.OrderedBound([-1, 1])
            ),
            mutation=mutation.Normal(
                # matching_type=rule.matching.OrderedBound([-1, 1]),
                sigma=1.22
            ),
            fitness_objs=[lambda r: r.error_],
            fitness_objs_labels=["Error"],  # infogain objective is added internally
        ),
        solution_composition=GeneticAlgorithm(),
        n_initial_rules=4,
    )

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=4,
        n_jobs=1,
        verbose=10,
        scoring=["r2", "neg_mean_squared_error"],
        return_estimator=True,
    )

    create_plot(scores)

    log_scores(scores)
