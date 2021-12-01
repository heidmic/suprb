import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle as apply_shuffle

from suprb2 import SupRB2
from suprb2 import rule
from suprb2.logging.combination import CombinedLogger
from suprb2.logging.mlflow import MlflowLogger
from suprb2.logging.stdout import StdoutLogger
from suprb2.optimizer.individual import ga
from suprb2.optimizer.rule import es
from suprb2.utils import check_random_state

plt.rcParams["figure.figsize"] = (8, 8)


def load_higdon_gramacy_lee(n_samples=1000, noise=0., shuffle=True, random_state=None):
    random_state_ = check_random_state(random_state)
    X = np.linspace(0, 20, num=n_samples)
    y = np.zeros(n_samples)
    y[X < 10] = np.sin(np.pi * X[X < 10] / 5) + 0.2 * np.cos(4 * np.pi * X[X < 10] / 5)
    y[X >= 10] = X[X >= 10] / 10 - 1
    y += random_state_.normal(scale=noise, size=n_samples)
    X = X.reshape((-1, 1))
    if shuffle:
        X, y = apply_shuffle(X, y, random_state=random_state)
    return X, y


if __name__ == '__main__':
    random_state = 42

    X, y = load_higdon_gramacy_lee(noise=0.1, random_state=random_state)
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    model = SupRB2(
        rule_generation=es.ES1xLambda(
            n_iter=100,
            operator='&',
            init=rule.initialization.MeanInit(fitness=rule.fitness.VolumeWu(alpha=0.05), ),
            mutation=es.mutation.HalfnormIncrease(sigma=0.1)
        ),
        individual_optimizer=ga.GeneticAlgorithm(
            n_iter=32,
            crossover=ga.crossover.Uniform(),
            selection=ga.selection.Tournament(),
        ),
        n_iter=4,
        n_rules=16,
        verbose=10,
        logger=CombinedLogger([('stdout', StdoutLogger()), ('mlflow', MlflowLogger())]),
        random_state=random_state,
    )

    scores = cross_validate(model, X, y, cv=4, n_jobs=1, verbose=10, scoring=['r2', 'neg_mean_squared_error'],
                            return_estimator=True)

    fig, axes = plt.subplots(2, 2)
    X_plot = np.linspace(X.min(), X.max(), 500).reshape((-1, 1))
    for ax, model in zip(axes.flatten(), scores['estimator']):
        pred = model.predict(X_plot)
        sorted_indices = np.argsort(X)
        ax.scatter(X, y, c='b', s=3, label='y_true')
        ax.plot(X_plot, pred, c='r', label='y_pred')

    fig.suptitle('Prediction on Higdon and Gramacy and Lee')
    fig.tight_layout()

    plt.show()
