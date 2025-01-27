import sklearn
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import Ridge


import suprb
from suprb.utils import check_random_state
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.solution.ga import GeneticAlgorithm
from suprb.wrapper import SupRBWrapper

from utils import log_scores


def load_higdon_gramacy_lee(n_samples=1000, noise=0, random_state=None):
    random_state_ = check_random_state(random_state)

    X = np.linspace(0, 20, num=n_samples)
    y = np.zeros(n_samples)

    y[X < 10] = np.sin(np.pi * X[X < 10] / 5) + 0.2 * np.cos(4 * np.pi * X[X < 10] / 5)
    y[X >= 10] = X[X >= 10] / 10 - 1

    y += random_state_.normal(scale=noise, size=n_samples)
    X = X.reshape((-1, 1))

    return sklearn.utils.shuffle(X, y, random_state=random_state)


if __name__ == "__main__":
    random_state = 42

    X, y = load_higdon_gramacy_lee(noise=0.1, random_state=random_state)

    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    # Comparable with examples/example_4.py
    basic_model = suprb.SupRB(random_state=33)
    
    model = SupRBWrapper(
        basic_model,
        print_config=True,
        ## RULE DISCOVERY ##
        rule_discovery=ES1xLambda(),
        rule_discovery__n_iter=10,
        rule_discovery__lmbda=16,
        rule_discovery__operator="+",
        rule_discovery__delay=150,
        rule_discovery__random_state=random_state,
        rule_discovery__n_jobs=1,
        rule_discovery__origin_generation=suprb.optimizer.rule.origin.Matching(),
        rule_discovery__origin_generation__use_elitist=True,
        rule_discovery__init=suprb.rule.initialization.MeanInit(),
        rule_discovery__init__model=Ridge(),
        rule_discovery__init__fitness=suprb.rule.fitness.VolumeWu(),
        rule_discovery__init__matching_type=suprb.rule.matching.OrderedBound(),
        rule_discovery__init__matching_type__bounds=[-1, 1],
        rule_discovery__mutation=suprb.optimizer.rule.mutation.HalfnormIncrease(),
        rule_discovery__mutation__sigma=0.1,
        rule_discovery__selection=suprb.optimizer.rule.selection.Fittest(),
        rule_discovery__acceptance=suprb.optimizer.rule.acceptance.Variance(),
        rule_discovery__acceptance__beta=0.1,
        rule_discovery__constraint=suprb.optimizer.rule.constraint.MinRange(),
        rule_discovery__constraint__min_range=1e-5,
        ## SOLUTION COMPOSITION ##
        solution_composition=GeneticAlgorithm(),
        solution_composition__n_iter=32,
        solution_composition__population_size=32,
        solution_composition__elitist_ratio=0.2,
        solution_composition__random_state=random_state,
        solution_composition__n_jobs=1,
        solution_composition__mutation=suprb.optimizer.solution.ga.mutation.BitFlips(),
        solution_composition__mutation__mutation_rate=0.1,
        solution_composition__crossover=suprb.optimizer.solution.ga.crossover.NPoint(),
        solution_composition__crossover__crossover_rate=0.91,
        solution_composition__crossover__n=3,
        solution_composition__selection=suprb.optimizer.solution.ga.selection.Tournament(),
        solution_composition__selection__k=6,
        solution_composition__init=suprb.solution.initialization.RandomInit(),
        solution_composition__init__mixing=suprb.solution.mixing_model.ErrorExperienceHeuristic(),
        solution_composition__init__fitness=suprb.solution.fitness.ComplexityWu(),
        solution_composition__init__p=0.6,
        solution_composition__archive=suprb.optimizer.solution.archive.Elitist(),
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
