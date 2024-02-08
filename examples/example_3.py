import sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.datasets import fetch_openml

import suprb
from suprb import SupRB
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.solution.ga import GeneticAlgorithm


def create_plot(scores):
    fig, axes = plt.subplots(2, 2)
    X_plot = np.linspace(X.min(), X.max(), 500).reshape((-1, 1))
    for ax, model in zip(axes.flatten(), scores['estimator']):
        pred = model.predict(X_plot)
        ax.scatter(X, y, c='b', s=3, label='y_true')
        ax.plot(X_plot, pred, c='r', label='y_pred')

    plt.savefig('result.png')


if __name__ == '__main__':
    random_state = 42

    data, _ = fetch_openml(name='Concrete_Data', version=1, return_X_y=True)
    data = data.to_numpy()

    X, y = data[:, :8], data[:, 8]
    X, y = sklearn.utils.shuffle(X, y, random_state=random_state)

    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    model = SupRB(rule_generation=ES1xLambda(n_iter=32,
                                             lmbda=16,
                                             operator='+',
                                             delay=150,
                                             random_state=random_state,
                                             n_jobs=1,
                                             origin_generation=suprb.optimizer.rule.origin.Matching(),
                                             init=suprb.rule.initialization.MeanInit(),
                                             mutation=suprb.optimizer.rule.mutation.HalfnormIncrease(),
                                             selection=suprb.optimizer.rule.selection.Fittest(),
                                             acceptance=suprb.optimizer.rule.acceptance.Variance(),
                                             constraint=suprb.optimizer.rule.constraint.MinRange()),
                  solution_composition=GeneticAlgorithm(n_iter=32,
                                                        population_size=32,
                                                        elitist_ratio=0.2,
                                                        random_state=random_state,
                                                        n_jobs=1,
                                                        mutation=suprb.optimizer.solution.ga.mutation.BitFlips(),
                                                        crossover=suprb.optimizer.solution.ga.crossover.NPoint(),
                                                        selection=suprb.optimizer.solution.ga.selection.Tournament(),
                                                        init=suprb.solution.initialization.RandomInit(),
                                                        archive=suprb.optimizer.solution.archive.Elitist()))

    scores = cross_validate(model, X, y, cv=4, n_jobs=1, verbose=10,
                            scoring=['r2', 'neg_mean_squared_error'],
                            return_estimator=True, fit_params={'cleanup': True})

    create_plot(scores)
