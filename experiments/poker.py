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
@click.option("-d", "--sample-size", type=click.IntRange(min=1), default=1000)
@click.option("-t", "--data-seed", type=click.IntRange(min=0), default=0)
def run_exp(seed, sample_size, data_seed):
    """
    Poker Hand Dataset
    https://archive.ics.uci.edu/ml/machine-learning-databases/poker/
    """
    print(f"Starting at {datetime.now().time()}")

    X_train, X_test, y_train, y_test = import_data(sample_size, data_seed)
    dimensions = X_train.shape[1]

    print(f"Samples generated. Starting training at {datetime.now().time()}")

    Config().rule_discovery["local_model"] = "log"
    configurations = create_configurations()
    for i in range(len(configurations)):
        mf.set_experiment(f"poker-{configurations[i]['name']}")
        opt_class_test_count = len([ d for d in configurations if d['name'] == configurations[i]['name'] ])
        Config().rule_discovery = { **Config().rule_discovery, **configurations[i] }

        with mf.start_run(run_name=f"{i % opt_class_test_count}"):
            mf.log_param("data_seed", data_seed)
            mf.log_param("sample_size", sample_size)
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


def import_data(sample_size, data_seed):
    Random().reseed(data_seed)

    data_train = pd.read_csv("datasets/poker/poker-hand-training-true.data", sep=',', header=None).values
    X_train, y_train = data_train[:,:-1], data_train[:,-1:]
    X_train, y_train = resample(X_train, y_train, n_samples=sample_size, random_state=Random().split_seed())

    data_test = pd.read_csv("datasets/poker/poker-hand-testing.data", sep=',', header=None).values
    X_test, y_test = data_test[:,:-1], data_test[:,-1:]
    X_test, y_test = resample(X_test, y_test, n_samples=sample_size, random_state=Random().split_seed())

    return (X_train, X_test, y_train, y_test)


def create_configurations():
    configurations = list()
    config = dict()

    for optimizer_type in ["ES_MLSP", "ES_CMA", "ES_OPL", "ES_ML"]:
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
