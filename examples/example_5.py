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


if __name__ == "__main__":
    random_state = 42
    nsga2 = NonDominatedSortingGeneticAlgorithm2(
        n_iter=32,
        population_size=32,
        random_state=random_state,
        n_jobs=1,
        sampler=BetaSolutionSampler(0.1, 0.1, projected=True),
    )
    ga = GeneticAlgorithm()
    ts = TwoStageSolutionComposition(
        algorithm_1=ga,
        algorithm_2=nsga2,
        switch_iteration=2,
    )
    sc_algos = (ts, nsga2)

    score_list = []
    time_list = []

    plt.rcParams.update(
        {
            "text.usetex": True,
        }
    )

    for sc in sc_algos:

        data, _ = fetch_openml(name="Concrete_Data", version=1, return_X_y=True)
        data = data.to_numpy()

        X, y = data[:, :8], data[:, 8]
        X, y = sklearn.utils.shuffle(X, y, random_state=random_state)

        X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
        y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

        suprb_iter = 32

        model = SupRB(
            n_iter=suprb_iter,
            rule_discovery=ES1xLambda(
                n_iter=32,
                lmbda=16,
                operator="+",
                delay=150,
                random_state=random_state,
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

        pareto_front = scores["estimator"][0].logger_.pareto_fronts_
        pareto_front = np.array(pareto_front[suprb_iter - 1])

        x = pareto_front[:, 0]
        y = pareto_front[:, 1]
        plt.step(x, y, color="black", linestyle="-", marker="x")
        hvs = scores["estimator"][0].logger_.metrics_["hypervolume"]
        hv = hvs[suprb_iter - 1]
        spreads = scores["estimator"][0].logger_.metrics_["spread"]
        spread = spreads[suprb_iter - 1]
        plt.title(f"$HV = {hv:.4f}, \Delta = {spread:.4f}$")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Complexity")
        plt.ylabel("Error")
        plt.show()
        print("Finished!")

    for t in time_list:
        print(f"Time taken: {t}")
