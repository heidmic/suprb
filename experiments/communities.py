from suprb2 import LCS
from suprb2.config import Config
from suprb2.random_gen import Random

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
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
def run_exp(seed, sample_size, data_seed):
    """
    Communities and Crime Data Set
    https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime
    """
    print(f"Starting at {datetime.now().time()}")

    X_train, X_test, y_train, y_test = import_data(sample_size, data_seed)
    dimensions = X_train.shape[1]

    print(f"Samples generated. Starting training at {datetime.now().time()}")

    configurations = create_configurations()
    for i in range(len(configurations)):
        mf.set_experiment(f"communities-{configurations[i]['name']}")
        opt_class_test_count = len([ d for d in configurations if d['name'] == configurations[i]['name'] ])
        Config().rule_discovery = { **Config().rule_discovery, **configurations[i] }

        with mf.start_run(run_name=f"{i % opt_class_test_count}"):
            mf.log_param("data_seed", data_seed)
            mf.log_param("sample_size", sample_size)
            mf.log_param("sample_dim", dimensions)
            mf.log_param("dataset", "communities")

            # we reset the seed here
            Random().reseed(seed)

            lcs = LCS(dimensions)
            lcs.fit(X_train, y_train)
            y_pred = lcs.predict(X_test)
            error = mean_squared_error(y_test, y_pred)

            mf.log_metric("RMSE", np.sqrt(error))
            print(f"Finished at {datetime.now().time()}. RMSE was {np.sqrt(error)}")


def import_data(sample_size, data_seed):
    Random().reseed(data_seed)

    data = pd.read_csv("datasets/communities/communities.data", sep=',', header=None, na_values=["?"]).values

    X, y = data[:,4:-1], data[:,-1].reshape(-1, 1)
    scale_X = MinMaxScaler(feature_range=(-1, 1))
    scale_X.fit(X)
    X = scale_X.transform(X)

    scale_y = StandardScaler()
    scale_y.fit(y)
    y = scale_y.transform(y)

    return train_test_split(X, y, train_size=sample_size, random_state=Random().split_seed())


def create_configurations():
    configurations = list()
    config = dict()

    for optimizer_type in ["ES_ML", "ES_OPL", "ES_MLSP","ES_CMA"]:
        config["name"] = optimizer_type
        for steps_per_step in [10, 100, 500, 1000]:
            config["steps_per_step"] = steps_per_step
            if optimizer_type == "ES_ML":
                for recombination_type in ["i", "d"]:
                    config["recombination"] = recombination_type
                    for replacement_type in ["+", ","]:
                        config["replacement"] = replacement_type
                        for mu in [10, 50, 100, 200, 500]:
                            config["mu"] = mu
                            for lmbd in [10, 50, 100, 200, 500]:
                                if lmbd > mu:
                                    break
                                else:
                                    config["lmbd"] = lmbd
                                    for rho in [10, 50, 100, 200, 500]:
                                        if rho > lmbd:
                                            break
                                        else:
                                            config["rho"] = rho
                                            for global_tau in [0.1, 0.25, 0.5]:
                                                config["global_tau"] = global_tau
                                                for local_tau in [0.1, 0.25, 0.5]:
                                                    config["local_tau"] = local_tau
                                                    configurations.append(config.copy())
            elif optimizer_type == "ES_MLSP":
                for lmbd in [20, 60, 100, 200, 520]:
                    config["lmbd"] = lmbd
                    config["mu"] = lmbd // 4
                    configurations.append(config.copy())
            else:
                for mu in [10, 50, 100, 200, 500]:
                    config["mu"] = mu
                    for lmbd in [10, 50, 100, 200, 500]:
                        config["lmbd"] = lmbd
                        configurations.append(config.copy())

    return configurations


if __name__ == '__main__':
    run_exp()
