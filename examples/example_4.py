import sklearn

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.datasets import fetch_openml
from sklearn.linear_model import Ridge

import suprb
from suprb import SupRB
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.solution.ga import GeneticAlgorithm

from utils import log_scores


if __name__ == "__main__":
    random_state = 42

    data, _ = fetch_openml(name="Concrete_Data", version=1, return_X_y=True)
    data = data.to_numpy()

    X, y = data[:, :8], data[:, 8]
    X, y = sklearn.utils.shuffle(X, y, random_state=random_state)

    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    model = SupRB(
        rule_discovery=ES1xLambda(
            n_iter=1000,
            lmbda=16,
            operator="&",
            delay=150,
            random_state=random_state,
            n_jobs=1,
            origin_generation=suprb.optimizer.rule.origin.Matching(use_elitist=True),
            init=suprb.rule.initialization.MeanInit(
                model=Ridge(),
                fitness=suprb.rule.fitness.VolumeWu(),
                matching_type=suprb.rule.matching.OrderedBound([-1, 1]),
            ),
            mutation=suprb.optimizer.rule.mutation.HalfnormIncrease(sigma=0.1),
            selection=suprb.optimizer.rule.selection.Fittest(),
            acceptance=suprb.optimizer.rule.acceptance.Variance(beta=0.1),
            constraint=suprb.optimizer.rule.constraint.MinRange(min_range=1e-6),
        ),
        solution_composition=GeneticAlgorithm(
            n_iter=32,
            population_size=32,
            elitist_ratio=0.2,
            random_state=random_state,
            n_jobs=1,
            mutation=suprb.optimizer.solution.ga.mutation.BitFlips(),
            crossover=suprb.optimizer.solution.ga.crossover.NPoint(n=2),
            selection=suprb.optimizer.solution.ga.selection.Tournament(k=6),
            init=suprb.solution.initialization.RandomInit(
                mixing=suprb.solution.mixing_model.ErrorExperienceHeuristic(),
                fitness=suprb.solution.fitness.ComplexityWu(),
                p=0.5,
            ),
            archive=suprb.optimizer.solution.archive.Elitist(),
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
    )

    log_scores(scores)
