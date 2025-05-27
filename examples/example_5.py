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

algo_names = {
    StrengthParetoEvolutionaryAlgorithm2: "SPEA2",
    NonDominatedSortingGeneticAlgorithm2: "NSGA-II",
    NonDominatedSortingGeneticAlgorithm3: "NSGA-III",
    TwoStageSolutionComposition: "TS",
}


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
    suprb_iter = 4
    sc_iter = 4

    spea2 = StrengthParetoEvolutionaryAlgorithm2(
        n_iter=sc_iter,
        population_size=32,
        sampler=BetaSolutionSampler(),
        early_stopping_delta=0,
        early_stopping_patience=10,
    )
    nsga2 = NonDominatedSortingGeneticAlgorithm2(
        n_iter=sc_iter,
        population_size=32,
        sampler=BetaSolutionSampler(),
        early_stopping_delta=0,
        early_stopping_patience=10,
    )
    nsga3 = NonDominatedSortingGeneticAlgorithm3(
        n_iter=sc_iter,
        population_size=32,
        sampler=BetaSolutionSampler(),
        early_stopping_delta=0,
        early_stopping_patience=10,
    )
    ga = GeneticAlgorithm(n_iter=sc_iter)
    ts = TwoStageSolutionComposition(
        algorithm_1=ga,
        algorithm_2=ga,
        switch_iteration=suprb_iter,
    )
    sc_algos = (nsga3, ts)
    logger_list = []
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
            rule_discovery=ES1xLambda(n_iter=1000),
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
        logger_list.append(scores["estimator"][0].logger_)
    print("Finished!")

    for t in time_list:
        print(f"Time taken: {t}")
    axes, plots = plt.subplots()
    for l in logger_list:
        ##### Plot Pareto Fronts #####
        pareto_front = l.pareto_fronts_
        pareto_front = np.array(pareto_front[suprb_iter - 1])
        hvs = l.metrics_["hypervolume"]
        hv = hvs[suprb_iter - 1]
        spreads = l.metrics_["spread"]
        spread = spreads[suprb_iter - 1]
        plot_pareto_front(pareto_front, f"$HV = {hv:.2f}, \Delta = {spread:.2f}$")
        plt.show()

        ##### Plot Hypervolume #####
