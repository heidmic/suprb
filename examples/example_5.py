import numpy as np
import sklearn
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.datasets import fetch_openml

from matplotlib import pyplot as plt
from sklearn.utils import Bunch

from suprb import SupRB
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.solution.nsga2 import NonDominatedSortingGeneticAlgorithm2
from suprb.optimizer.solution.sampler import BetaSolutionSampler, DiversitySolutionSampler
from suprb.optimizer.solution.spea2 import StrengthParetoEvolutionaryAlgorithm2
from suprb.optimizer.solution.nsga3 import NonDominatedSortingGeneticAlgorithm3
from suprb.optimizer.solution.ga import GeneticAlgorithm
from suprb.optimizer.solution.ts import TwoStageSolutionComposition
from suprb.logging.multi_objective import MOLogger

from utils import log_scores

import time


def plot_pareto_front(pareto_front: np.ndarray, title: str):
    x = pareto_front[:, 0]
    y = pareto_front[:, 1]
    plt.step(x, y, linestyle="-", marker="x")
    plt.title(title)
    plt.xlabel("Complexity")
    plt.ylabel("Error")
    plt.xlim(0, 1)
    plt.ylim(0, 1)


if __name__ == "__main__":
    random_state = 42

    suprb_iter = 32

    spea2 = StrengthParetoEvolutionaryAlgorithm2(
        n_iter=128,
        population_size=32,
        sampler=BetaSolutionSampler(),
        early_stopping_delta=0,
        early_stopping_patience=10,
    )
    nsga2 = NonDominatedSortingGeneticAlgorithm2(
        n_iter=128,
        population_size=32,
        sampler=BetaSolutionSampler(),
        early_stopping_delta=0,
        early_stopping_patience=10,
    )
    nsga3 = NonDominatedSortingGeneticAlgorithm3(
        n_iter=128,
        population_size=32,
        sampler=BetaSolutionSampler(),
        early_stopping_delta=0,
        early_stopping_patience=10,
    )
    ga = GeneticAlgorithm()
    ts = TwoStageSolutionComposition(
        algorithm_1=ga,
        algorithm_2=nsga3,
        switch_iteration=8,
    )
    sc_algos = (nsga2, nsga3, spea2, ts)

    score_list = []
    time_list = []

    plt.rcParams.update(
        {
            "text.usetex": True,
        }
    )

    for i, sc in enumerate(sc_algos):

        data, _ = fetch_openml(name="Concrete_Data", version=1, return_X_y=True)
        data = data.to_numpy()

        X, y = data[:, :8], data[:, 8]
        X, y = sklearn.utils.shuffle(X, y, random_state=random_state)

        X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
        y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

        model = SupRB(
            n_iter=suprb_iter,
            rule_discovery=ES1xLambda(
                n_iter=32,
                lmbda=16,
                operator="+",
                delay=150,
                n_jobs=1,
            ),
            solution_composition=sc,
            logger=MOLogger(),
            random_state=random_state,
        )

        start_time = time.time()
        scores = cross_validate(
            model,
            X,
            y,
            cv=2,
            n_jobs=2,
            verbose=10,
            scoring=["r2", "neg_mean_squared_error"],
            return_estimator=True,
            fit_params={"cleanup": True},
        )
        end_time = time.time()
        time_list.append(end_time - start_time)

        log_scores(scores)

        score_list.append(scores)

        try:
            pareto_front = scores["estimator"][0].logger_.pareto_fronts_
            pareto_front = np.array(pareto_front[suprb_iter - 1])
            hvs = scores["estimator"][0].logger_.metrics_["hypervolume"]
            hv = hvs[suprb_iter - 1]
            spreads = scores["estimator"][0].logger_.metrics_["spread"]
            spread = spreads[suprb_iter - 1]
            plot_pareto_front(pareto_front, f"$HV = {hv:.2f}, SP = {spread:.2f}$")
        except Exception:
            elitist_error = scores["estimator"][0].logger_.metrics_["elitist_error"]
            elitist_complexity = scores["estimator"][0].logger_.metrics_["elitist_complexity"]
            elitist_error = np.array(list(elitist_error.values()))
            elitist_complexity = np.array(list(elitist_complexity.values()))
            pseudo_accuracy = 1 - np.exp(-2 * elitist_error)
            c_norm = elitist_complexity / (4 * suprb_iter)

            plt.plot(c_norm, pseudo_accuracy, marker="o", linestyle="-")
            for idx, (x, y) in enumerate(zip(c_norm, pseudo_accuracy)):
                plt.text(x, y, str(idx), fontsize=8, ha="right", va="bottom")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
    plt.show()
    print("Finished!")

    for t in time_list:
        print(f"Time taken: {t}")
