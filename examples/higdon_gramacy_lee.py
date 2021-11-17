import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from suprb2 import SupRB2
from suprb2.optimizer import rule, individual
from suprb2.optimizer.individual import ga
from suprb2.optimizer.rule import es
from suprb2.utils import check_random_state

sns.set_theme(style='darkgrid')
sns.set(rc={'figure.figsize': (9, 9)})


def load_higdon_gramacy_lee(n_samples=1000, noise=0., random_state=None):
    random_state_ = check_random_state(random_state)
    X = np.linspace(0, 20, num=n_samples)
    y = np.zeros(n_samples)
    y[X < 10] = np.sin(np.pi * X[X < 10] / 5) + 0.2 * np.cos(4 * np.pi * X[X < 10] / 5)
    y[X >= 10] = X[X >= 10] / 10 - 1
    y += random_state_.normal(scale=noise, size=n_samples)
    X = X.reshape((-1, 1))
    return X, y


if __name__ == '__main__':
    random_state = 42

    X, y = load_higdon_gramacy_lee(noise=0.1)
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    model = SupRB2(
        rule_generation=rule.es.ES1xLambda(
            fitness=rule.fitness.VolumeWu(alpha=0.05),
            init=rule.initialization.MeanInit(),
            mutation=rule.es.mutation.HalfnormIncrease(sigma=0.1)
        ),
        individual_optimizer=individual.ga.GeneticAlgorithm(
            n_iter=32,
            crossover=individual.ga.crossover.Uniform(),
        ),
        n_iter=16,
        n_rules=4,
        progress_bar=True,
        random_state=random_state,
    )

    scores = cross_validate(model, X, y, cv=9, n_jobs=1, verbose=10, scoring=['r2', 'neg_mean_squared_error'],
                            return_estimator=True)

    fig, axes = plt.subplots(3, 3)
    for ax, model in zip(axes.flatten(), scores['estimator']):
        pred = model.predict(X)
        ax.scatter(X, y, c='b', s=3, label='y_true')
        ax.plot(X, pred, c='r', label='y_pred')

    fig.suptitle('Prediction on Higdon and Gramacy and Lee')
    fig.tight_layout()

    plt.show()
