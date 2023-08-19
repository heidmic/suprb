import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle as apply_shuffle

from suprb import SupRB
from suprb import rule
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer import rule as rule_opt
from suprb.optimizer.solution import ga
from suprb.utils import check_random_state
from suprb.optimizer.rule.mutation import HalfnormIncrease
from sklearn.linear_model import Ridge
from suprb.optimizer.rule.ns.novelty_calculation import NoveltyFitnessBiased

from suprb.optimizer.rule import ns, origin


def load_higdon_gramacy_lee(n_samples=1000, noise=0., shuffle=True, random_state=None):
    random_state_ = check_random_state(random_state)
    X = np.linspace(0, 20, num=n_samples)
    y = np.zeros(n_samples)
    y[X < 10] = np.sin(np.pi * X[X < 10] / 5) + 0.2 * np.cos(4 * np.pi * X[X < 10] / 5)
    y[X >= 10] = X[X >= 10] / 10 - 1
    y += random_state_.normal(scale=noise, size=n_samples)
    X = X.reshape((-1, 1))
    if shuffle:
        # Note that we have to supply `random_state` (the parameter) to `apply_shuffle` here,
        # and not `random_state_` (the `np.random.Generator` instance)
        # because `sklearn.utils.check_random_state()` does currently not support `np.random.Generator`.
        # See https://github.com/scikit-learn/scikit-learn/issues/16988 for the current status.
        # Our `suprb.utils.check_random_state()` can handle `np.random.Generator`.
        X, y = apply_shuffle(X, y, random_state=random_state)
    return X, y


if __name__ == '__main__':
    random_state = 42

    # Prepare the data
    X, y = load_higdon_gramacy_lee(noise=0.1, random_state=random_state)
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    # Prepare the model
    model = SupRB(
        rule_generation=ns.NoveltySearch(
            init=rule.initialization.MeanInit(fitness=rule.fitness.VolumeWu(),
                                              model=Ridge(alpha=0.01,
                                                          random_state=random_state)),
            origin_generation=origin.SquaredError(),
            mutation=HalfnormIncrease(),
            novelty_calculation=NoveltyFitnessBiased()
        ),
        solution_composition=ga.GeneticAlgorithm(n_iter=32, population_size=32),
        n_iter=32,
        n_rules=8,
        verbose=10,
        logger=CombinedLogger(
            [('stdout', StdoutLogger()), ('default', DefaultLogger())]),
    )

    # Do cross-validation
    scores = cross_validate(model, X_train, y_train, cv=4, n_jobs=1, verbose=10,
                            scoring=['r2', 'neg_mean_squared_error'],
                            return_estimator=True, fit_params={'cleanup': True})
    models = scores['estimator']

    # Plot the prediction
    plt.rcParams["figure.figsize"] = (8, 8)
    fig, axes = plt.subplots(2, 2)
    X_plot = np.linspace(X.min(), X.max(), 500).reshape((-1, 1))
    for ax, model in zip(axes.flatten(), scores['estimator']):
        pred = model.predict(X_plot)
        sorted_indices = np.argsort(X)
        ax.scatter(X, y, c='b', s=3, label='y_true')
        ax.plot(X_plot, pred, c='r', label='y_pred')

    fig.suptitle('Prediction on Higdon and Gramacy and Lee')
    fig.tight_layout()

    # Log the results to mlflow
    mlflow.set_experiment("Higdon & Gramacy & Lee")

    with mlflow.start_run(run_name="Cross-Validation"):
        mlflow.log_param("cv", len(models))
        for i, estimator in enumerate(models):
            with mlflow.start_run(run_name=f"Fold {i}", nested=True):
                logger = estimator.logger_.loggers_[1][1]

                # Log model parameters
                mlflow.log_params(logger.params_)

                # Log fitting metrics
                for key, values in logger.metrics_.items():
                    for step, value in values.items():
                        mlflow.log_metric(key=key, value=value, step=step)

                # Log test metrics
                mlflow.log_metric("test_score", estimator.score(X_test, y_test))

        # Add the figure as artifact
        mlflow.log_figure(fig, 'predictions.png')
