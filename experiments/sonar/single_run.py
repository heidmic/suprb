from suprb2 import LCS
from suprb2.random_gen import Random

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from datetime import datetime
import numpy as np
import pandas as pd
import mlflow as mf
import click

@click.command()
@click.option("-s", "--seed", type=click.IntRange(min=0), default=0)
@click.option("-t", "--data-seed", type=click.IntRange(min=0), default=0)
@click.option("-c", "--config-path", default="suprb2/config.py")
@click.option("-n", "--run-name", default="")
def run_exp(seed, data_seed, config_path, run_name):
    """
    Connectionist Bench (Sonar, Mines vs. Rocks) Data Set
    https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)
    """
    print(f"Starting at {datetime.now().time()}")

    X_train, X_test, y_train, y_test = import_data(data_seed)
    dimensions = X_train.shape[1]

    # Import the configurations for this run
    print(f"Configurations directory: {config_path}")
    module_path = config_path.replace("/", ".")[:-3]
    module = __import__(module_path, fromlist=["Config"])
    config_class = getattr(module, "Config")
    config = config_class()

    print(f"Samples generated. Starting training at {datetime.now().time()}")

    mf.set_experiment(f"Test with sonar dataset")
    with mf.start_run(run_name=run_name):
        mf.log_param("data_seed", data_seed)
        mf.log_param("sample_dim", dimensions)
        mf.log_param("dataset", "sonar")

        # we reset the seed here
        Random().reseed(seed)

        lcs = LCS(dimensions, config)
        lcs.fit(X_train, y_train)
        y_pred = lcs.predict(X_test)
        macro_f1_score = f1_score(y_test, np.rint(y_pred), average='macro')

        mf.log_metric("final macro f1 score", macro_f1_score)
        print(f"Finished at {datetime.now().time()}. Macro F1 Score was {macro_f1_score}")


def import_data(data_seed):
    Random().reseed(data_seed)

    data = pd.read_csv("datasets/sonar/sonar.csv", sep=',', header=None).values

    X, y = data[:,:-1], data[:,-1]
    scale_X = MinMaxScaler(feature_range=(-1, 1))
    scale_X.fit(X)
    X = scale_X.transform(X)

    encode_y = LabelEncoder()
    encode_y.fit(y)
    y = encode_y.transform(y)

    return train_test_split(X, y, random_state=Random().split_seed())


if __name__ == '__main__':
    run_exp()
