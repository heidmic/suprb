from problems import amgauss, make_problem
from suprb2 import LCS
from suprb2.random_gen import Random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt


def f(X):
    return 0.75*X**3-5*X**2+4*X+12


def plot_results(X, y_test, y_pred, elitist=None):
    fig, ax = plt.subplots()
    plt.scatter(X, y_test, marker='^', c='red', label='test')
    plt.scatter(X, y_pred, marker='o', c='blue', label='pred')
    plt.xlabel('x')
    plt.ylabel('y')

    if elitist is not None:
        import matplotlib.colors as mcolors
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle
        colors = list(mcolors.CSS4_COLORS.values())
        size = np.max(y_test) - np.min(y_test)
        per_cl = size / len(elitist.classifiers)
        for i in range(len(elitist.classifiers)):
            plt.axvline(elitist.classifiers[i].lowerBounds[0], color=colors[i], lw=1)
            plt.axvline(elitist.classifiers[i].upperBounds[0], color=colors[i], lw=1)
            ax.add_patch(Rectangle((elitist.classifiers[i].lowerBounds[0], np.min(y_test)+i*per_cl), elitist.classifiers[i].upperBounds[0]-elitist.classifiers[i].lowerBounds[0], per_cl, fill=False, linewidth=2, edgecolor=colors[i], hatch='/'))

    plt.legend()

    plt.show()


def plot_perfrecords(recorder, ignore=[]):
    g = np.arange(0, len(list(recorder.values())[0]), 1)
    for i in range(len(recorder.keys())):
        if list(recorder.keys())[i] in ignore:
            continue
        plt.scatter(g, list(recorder.values())[i], marker=i, label=list(recorder.keys())[i])

    plt.xlabel('Generations')
    plt.ylabel('value')
    plt.legend()

    plt.show()


def plot_error_complexity(recorder):
    g = np.arange(0, len(recorder.elitist_val_error), 1)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('val_error', color=color)
    ax1.plot(g, recorder.elitist_val_error, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('complexity', color=color)  # we already handled the x-label with ax1
    ax2.plot(g, recorder.elitist_complexity, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


if __name__ == '__main__':
    print(f"Starting at {datetime.now().time()}")
    n = 10000

    """prob = make_problem("amgauss", 1)

    X, y = prob.generate(n)

    y = np.reshape(y, (-1, 1))
    
    xdim = prob.xdim + prob.adim"""

    Random().seed(0)

    X = Random().random.uniform(-2.5, 7, (n, 1))
    y = f(X)
    xdim = 1

    scale_X = MinMaxScaler(feature_range=(-1, 1))
    scale_X.fit(X)
    X = scale_X.transform(X)

    scale_y = StandardScaler()
    scale_y.fit(y)
    y = scale_y.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=Random().split_seed())

    print(f"Samples generated. Starting training at {datetime.now().time()}")

    lcs = LCS(xdim=xdim, pop_size=50, ind_size=10, cl_min_range=0.2, generations=50,
              fitness="pseudo-BIC")

    lcs.fit(X_train, y_train)

        # we reset the seed here
        Random().seed(0)
    y_pred = lcs.predict(X_test)

    error = mean_squared_error(y_test, y_pred)

    plot_results(X_test, y_test, y_pred, lcs.elitist)

    plot_perfrecords(lcs.perf_recording.__dict__, ["elitist_complexity", "elitist_val_error"])
    plot_error_complexity(lcs.perf_recording)

    print(f"Finished at {datetime.now().time()}. RMSE was {np.sqrt(error)}")

    pass

