from suprb2 import LCS
from suprb2.config import Config
from suprb2.random_gen import Random

from sklearn.utils import resample
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder

from datetime import datetime
import numpy as np
import pandas as pd
import mlflow as mf
import click

@click.command()
@click.option("-s", "--seed", type=click.IntRange(min=0), default=0)
@click.option("-t", "--data-seed", type=click.IntRange(min=0), default=0)
@click.option("-n", "--run-name", default="")
def run_exp(seed, data_seed, run_name):
    """
    Poker Hand Dataset
    https://archive.ics.uci.edu/ml/machine-learning-databases/poker/
    """
    print(f"Starting at {datetime.now().time()}")

    X_train, X_test, y_train, y_test = import_data(data_seed)
    dimensions = X_train.shape[1]

    print(f"Samples generated. Starting training at {datetime.now().time()}")

    Config().classifier["local_model"] = "logistic_regression"
    Config().solution_creation["fitness"] = "pseudo-BIC"
    Config()["default_error"] = 0.9

    mf.set_experiment("Test with poker dataset")
    with mf.start_run(run_name=run_name):
        mf.log_param("data_seed", data_seed)
        mf.log_param("sample_size", X_train.shape[0])
        mf.log_param("sample_dim", dimensions)
        mf.log_param("dataset", "poker")

        # we reset the seed here
        Random().reseed(seed)

        lcs = LCS(dimensions)
        lcs.fit(X_train, y_train)
        y_pred = lcs.predict(X_test)
        score = f1_score(y_test, y_pred, average="macro")

        mf.log_metric("F1-Score", score)
        print(f"Finished at {datetime.now().time()}. F1-Score was {score}")


def import_data(data_seed):
    Random().reseed(data_seed)
    scale_X = OneHotEncoder(sparse=False)

    data_train = pd.read_csv("datasets/poker/poker-hand-training-true.data", sep=',', header=None).values
    X, y_train = data_train[:,:-1], data_train[:,-1:].flatten()
    scale_X.fit(X)
    X_train = scale_X.transform(X)

    data_test = pd.read_csv("datasets/poker/poker-hand-testing.data", sep=',', header=None).values
    X, y_test = data_test[:,:-1], data_test[:,-1:].flatten()
    scale_X.fit(X)
    X_test = scale_X.transform(X)

    return (X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    run_exp()
