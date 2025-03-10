from ucimlrepo import fetch_ucirepo 
import sklearn
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.utils import shuffle

import suprb
from sklearn.compose import make_column_transformer
from suprb.utils import check_random_state
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.rule.acceptance import MaxError
from suprb.optimizer.solution.ga import GeneticAlgorithm
from suprb.wrapper import SupRBWrapper

from utils import log_scores


if __name__ == '__main__':
    random_state = 125
    local_model = LogisticRegression(penalty='l1', C=0.1, random_state=random_state, solver='saga', tol=0.001, max_iter=1000)
    #CLASSIFICATION
    # fetch dataset 
    iris = fetch_ucirepo(id=53)
    X = iris.data.features.to_numpy()
    y = iris.data.targets.to_numpy()
    X, y = shuffle(X, y, random_state=random_state)
    unique = np.unique(y)
    toNum = dict(zip(unique, range(1, len(unique)+1)))
    # Conversion of tragets to int required for mixing
    # Similiar to sklearn.preprocessing.OrdinalEncoder
    y = [toNum[x[0]] for x in y]
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    # Comparable with examples/example_2.py
    model = SupRBWrapper(print_config=True,
                        #n_iter=10,
                         ## RULE GENERATION ##
                         rule_generation=ES1xLambda(),
                         rule_generation__n_iter=16,
                         rule_generation__lmbda=16,
                         rule_generation__operator='+',
                         rule_generation__delay=10,
                         rule_generation__random_state=random_state,
                         rule_generation__n_jobs=4,
                         rule_generation__init__model=local_model, 

                         ## SOLUTION COMPOSITION ##
                         solution_composition=GeneticAlgorithm(),
                         solution_composition__init__mixing=suprb.solution.mixing_model.ErrorExperienceClassification(),
                         solution_composition__n_iter=32,
                         solution_composition__population_size=32,
                         solution_composition__elitist_ratio=0.2,
                         solution_composition__random_state=random_state,
                         solution_composition__n_jobs=4)

    scores = cross_validate(model, X, y, cv=4, n_jobs=4, verbose=10,
                            scoring=['accuracy'],
                            return_estimator=True, fit_params={'cleanup': True})

    log_scores(scores)
