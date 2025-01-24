import sklearn

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.datasets import fetch_openml

from suprb import SupRB
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.solution.ga import GeneticAlgorithm

from utils import log_scores
import cProfile
import pstats
from line_profiler import LineProfiler
from suprb.rule.matching import OrderedBound
from suprb.optimizer.rule.mutation import RuleMutation
from suprb.optimizer.rule.mutation import HalfnormIncrease
from suprb.rule import Rule


def start_test():
    random_state = 42

    data, _ = fetch_openml(name="Concrete_Data", version=1, return_X_y=True)
    data = data.to_numpy()

    X, y = data[:, :8], data[:, 8]
    X, y = sklearn.utils.shuffle(X, y, random_state=random_state)

    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    model = SupRB(
        rule_discovery=ES1xLambda(
            n_iter=8,
            lmbda=8,
            operator="+",
            delay=2,
            random_state=random_state,
            n_jobs=1,
        ),
        solution_composition=GeneticAlgorithm(
            n_iter=4,
            population_size=4,
            elitist_ratio=0.2,
            random_state=random_state,
            n_jobs=1,
        ),
    )

    scores = cross_validate(
        model,
        X,
        y,
        cv=4,
        n_jobs=1,
        verbose=10,
        scoring=["r2", "neg_mean_squared_error"],
        return_estimator=True,
        fit_params={"cleanup": True},
    )

    log_scores(scores)


if __name__ == "__main__":
    cProfile.run('start_test()', 'output.prof')

    with open("profiler_results.txt", "w") as f:
        ps = pstats.Stats('output.prof', stream=f)
        ps.sort_stats('cumulative').print_stats()
        ps.strip_dirs().sort_stats('cumulative').print_stats('_catch_errors')



    # profiler = LineProfiler()
    # profiler.add_function(RuleMutation.__call__)
    # profiler.add_function(Rule.fit)
    # profiler.add_function(SupRB._catch_errors)
    # profiler.add_function(HalfnormIncrease.ordered_bound)

    # profiler.run('start_test()')
    # output_file = "line_profiler_results.txt"
    # with open(output_file, "w") as f:
    #     profiler.print_stats(stream=f)

