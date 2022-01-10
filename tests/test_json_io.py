import numpy as np
import os

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
import suprb2.json_io as json_io


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


def setup():
    random_state = 42

    X, y = load_higdon_gramacy_lee(noise=0.1, random_state=random_state)
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    model = SupRB2(
        rule_generation=es.ES1xLambda(
            n_iter=2,
            operator='&',
            init=rule.initialization.MeanInit(fitness=rule.fitness.VolumeWu(alpha=0.05)),
            mutation=es.mutation.HalfnormIncrease(sigma=0.1)
        ),
        individual_optimizer=ga.GeneticAlgorithm(
            n_iter=1,
            crossover=ga.crossover.Uniform(),
            selection=ga.selection.Tournament(),
        ),
        n_iter=1,
        n_rules=5,
        verbose=1,
        logger=CombinedLogger([('stdout', StdoutLogger()), ('mlflow', MlflowLogger())]),
        random_state=random_state,
    )

    return model, X, y


class TestJsonIO:
    def test_save_load_config(self):
        model, X, y = setup()

        scores = cross_validate(model, X, y, cv=4, n_jobs=1, verbose=1, scoring=['r2', 'neg_mean_squared_error'],
                                return_estimator=True)

        model = scores["estimator"][0]
        json_io_params = model.get_params()

        original_config = json_io._save_config(model)
        model = json_io._load_config(original_config["config"])
        model_params = model.get_params()

        for key in json_io_params:
            assert(type(json_io_params[key]) == type(model_params[key]))

    def test_smoke_test(self):
        model, X, y = setup()

        X_predict = np.linspace(X.min(), X.max(), 500).reshape((-1, 1))

        scores = cross_validate(model, X, y, cv=4, n_jobs=1, verbose=1, scoring=['r2', 'neg_mean_squared_error'],
                                return_estimator=True)

        model = scores["estimator"][0]
        original_prediction = model.predict(X_predict)

        json_io.dump(model, "save_state.json")
        model = json_io.load("save_state.json")
        loaded_prediction = model.predict(X_predict)

        assert((original_prediction == loaded_prediction).all())
        os.remove("save_state.json")
