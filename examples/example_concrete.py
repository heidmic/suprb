import sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_validate
from ucimlrepo import fetch_ucirepo
from suprb import SupRB
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.solution.ga import GeneticAlgorithm
from utils import log_scores




if __name__ == "__main__":
    random_state = 42
    # Dataset https://doi.org/10.24432/C5PK67
    concrete_data = fetch_ucirepo(id=165)

    X, y = concrete_data.data.features, concrete_data.data.targets
    X = X.to_numpy()
    y = y.to_numpy()
    X, y = sklearn.utils.shuffle(X, y, random_state=random_state)

    # Normalize features and target
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(np.array(y).reshape(-1, 1)).reshape((-1,))

    # Define model
    model = SupRB(
        rule_discovery=ES1xLambda(
            n_iter=32,
            lmbda=16,
            operator="+",
            delay=150,
            random_state=random_state,
            n_jobs=1,
        ),
        solution_composition=GeneticAlgorithm(
            n_iter=32,
            population_size=32,
            elitist_ratio=0.2,
            random_state=random_state,
            n_jobs=1,
        ),
    )

    # Cross-validation
    scores = cross_validate(
        model,
        X,
        y,
        cv=4,
        n_jobs=32,
        verbose=10,
        scoring=["r2", "neg_mean_squared_error"],
        return_estimator=True,
        fit_params={"cleanup": True},
    )

    log_scores(scores)
