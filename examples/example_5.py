import numpy as np
import sklearn

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.datasets import fetch_openml

from matplotlib import pyplot as plt

from suprb import SupRB
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.solution.nsga2 import NonDominatedSortingGeneticAlgorithm2
from suprb.optimizer.solution.spea2 import StrengthParetoEvolutionaryAlgorithm2
from suprb.logging.multi_objective import MOLogger

from utils import log_scores


if __name__ == "__main__":
    random_state = 42

    data, _ = fetch_openml(name="Concrete_Data", version=1, return_X_y=True)
    data = data.to_numpy()

    X, y = data[:, :8], data[:, 8]
    X, y = sklearn.utils.shuffle(X, y, random_state=random_state)

    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    sc_algos = (NonDominatedSortingGeneticAlgorithm2, StrengthParetoEvolutionaryAlgorithm2)

    score_list = []

    plt.rcParams.update({
        "text.usetex": True,
    })

    for sc in sc_algos:
        random_state = 42

        data, _ = fetch_openml(name="Concrete_Data", version=1, return_X_y=True)
        data = data.to_numpy()

        X, y = data[:, :8], data[:, 8]
        X, y = sklearn.utils.shuffle(X, y, random_state=random_state)

        X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
        y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

        sc_algos = (NonDominatedSortingGeneticAlgorithm2, StrengthParetoEvolutionaryAlgorithm2)

        model = SupRB(
            rule_discovery=ES1xLambda(
                n_iter=32,
                lmbda=16,
                operator="+",
                delay=150,
                random_state=random_state,
                n_jobs=1,
            ),
            solution_composition=sc(
                n_iter=32,
                population_size=32,
                random_state=random_state,
                n_jobs=1,
            ),
            logger=MOLogger(),
            random_state=random_state,
        )

        scores = cross_validate(
            model,
            X,
            y,
            cv=2,
            n_jobs=1,
            verbose=10,
            scoring=["r2", "neg_mean_squared_error"],
            return_estimator=True,
            fit_params={"cleanup": True},
        )

        log_scores(scores)

        score_list.append(scores)

        pareto_front = scores["estimator"][0].logger_.pareto_front_
        pareto_front = np.array([solution.fitness_ for solution in pareto_front])

        x = pareto_front[:, 1]
        y = pareto_front[:, 0]
        plt.step(x, y, color="black", linestyle="-", marker="x")
        hv = scores["estimator"][0].logger_.metrics_["hypervolume"]
        spread = scores["estimator"][0].logger_.metrics_["spread"]
        plt.title(f"$HV = {hv}, \Delta = {spread}$")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Complexity")
        plt.ylabel("Error")
        plt.show()
        print("Finished!")
