from suprb2 import LCS
from suprb2.config import Config
from suprb2.random_gen import Random

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from datetime import datetime
import numpy as np
import pandas as pd
import mlflow as mf
import click

@click.command()
@click.option("-s", "--seed", type=click.IntRange(min=0), default=0)
@click.option("-d", "--sample-size", type=click.IntRange(min=1), default=1000)
@click.option("-t", "--data-seed", type=click.IntRange(min=0), default=0)
@click.option("-o", "--optimizer")
@click.option('-m', "--mu", type=click.IntRange(min=1))
@click.option('-l', "--lmbd", type=click.IntRange(min=1))
@click.option('-r', "--rho", type=click.IntRange(min=1))
@click.option('-sps', "--steps-per-step", type=click.IntRange(min=1))
@click.option('-rec', "--recombination")
@click.option('-lt', "--local-tau", type=float)
@click.option('-gt', "--global-tau", type=float)
@click.option('-rep', "--replacement")
def run_exp(seed, optimizer, sample_size, data_seed, mu, lmbd, rho, steps_per_step, recombination, local_tau, global_tau, replacement):
    print(f"Starting at {datetime.now().time()}")

    n = sample_size # Probably a good idea to try different samples...

    """Communities and Crime Data Set
    https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime"""

    # Setting relevant hyper parameters
    Config().rule_discovery['name'] = optimizer
    Config().rule_discovery['mu'] = mu
    Config().rule_discovery['lmbd'] = lmbd
    Config().rule_discovery['rho'] = rho
    Config().rule_discovery['steps_per_step'] = steps_per_step
    Config().rule_discovery['recombination'] = recombination
    Config().rule_discovery['local_tau'] = local_tau
    Config().rule_discovery['global_tau'] = global_tau
    Config().rule_discovery['replacement'] = replacement

    Random().reseed(data_seed)

    data = pd.read_csv("datasets/communities/communities.data", sep=',', header=None, na_values=["?"]).values
    dimensions = data.shape[1]
    X, y = data[:,4:-1], data[:,-1].reshape(-1, 1)

    scale_X = MinMaxScaler(feature_range=(-1, 1))
    scale_X.fit(X)
    X = scale_X.transform(X)

    scale_y = StandardScaler()
    scale_y.fit(y)
    y = scale_y.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=Random().split_seed())

    print(f"Samples generated. Starting training at {datetime.now().time()}")

    mf.set_experiment("Tests with Communities and Crime Dataset")

    with mf.start_run():
        mf.log_param("data_seed", data_seed)
        mf.log_param("sample_size", sample_size)
        mf.log_param("sample_dim", dimensions)
        mf.log_param("dataset", "Communities and Crime Dataset")

        # we reset the seed here
        Random().reseed(seed)

        lcs = LCS(dimensions)

        lcs.fit(X_train, y_train)

        y_pred = lcs.predict(X_test)

        error = mean_squared_error(y_test, y_pred)

        mf.log_metric("RMSE", np.sqrt(error))
        print(f"Finished at {datetime.now().time()}. RMSE was {np.sqrt(error)}")

    pass


if __name__ == '__main__':
    run_exp()
