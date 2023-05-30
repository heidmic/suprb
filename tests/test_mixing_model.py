import unittest
import numpy as np

import suprb.solution.mixing_model as mixing_model

from suprb.rule import Rule
from suprb.rule.matching import OrderedBound
from suprb.rule.fitness import VolumeWu
from sklearn.base import RegressorMixin


import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle as apply_shuffle

from suprb import SupRB
from suprb import rule
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer import rule as rule_opt
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import es, rs
from suprb.utils import check_random_state
from suprb.optimizer.rule.mutation import HalfnormIncrease
from suprb.solution.initialization import RandomInit
import suprb.solution.mixing_model as mixing_model

random_state = 42

model = SupRB(
    rule_generation=es.ES1xLambda(
        operator='&',
        origin_generation=rule_opt.origin.Matching(),
        init=rule.initialization.MeanInit(
            fitness=rule.fitness.VolumeWu(alpha=0.05)),
        mutation=HalfnormIncrease(sigma=0.1),
    ),
    solution_composition=ga.GeneticAlgorithm(
        n_iter=1,
        crossover=ga.crossover.Uniform(),
        selection=ga.selection.Tournament(),
        init=RandomInit(mixing=mixing_model.NBestFitness(
            random_state=random_state, rule_amount=4))
    ),
    n_iter=1,
    n_rules=4,
    verbose=10,
    logger=CombinedLogger(
        [('stdout', StdoutLogger()), ('default', DefaultLogger())]),
    random_state=random_state,
)


class TestMixingModel(unittest.TestCase):

    def create_rule(self, fitness, experience, error):
        rule = Rule(match=OrderedBound(np.array([[-1, 1]])),
                    input_space=[-1.0, 1.0],
                    model=model,
                    fitness=VolumeWu)

        rule.fitness_ = fitness
        rule.experience_ = experience
        rule.error_ = error

        return rule

    def disabled_test_base_mixing_model(self):
        random_state = 42
        rule_amount = 1
        subpopulation = []
        X = np.linspace(0, 20, num=50)
        X = X.reshape((-1, 1))

        model.fit(X, X)

        rule = Rule(match=matching,
                    input_space=[-1.0, 1.0], model=model, fitness=VolumeWu)
        rule.fitness_ = 0.1
        rule.experience_ = 1
        rule.error_ = 0.01

        subpopulation.append(rule)

        rule2 = Rule(match=matching,
                     input_space=[-1.0, 1.0], model=model, fitness=VolumeWu)
        rule2.fitness_ = 0.5
        rule.experience_ = 1
        rule.error_ = 0.01

        subpopulation.append(rule)

        mixing = mixing_model.NRandom(random_state, rule_amount)

        result = mixing(X, subpopulation)

        print(result)

    def test_NBestFitness_filter_subpopulation(self):
        random_state = 42
        rule_amount = 4
        subpopulation = []

        X = np.linspace(0, 20, num=50)
        X = X.reshape((-1, 1))

        subpopulation.append(self.create_rule(0.8, 1, 0.01))
        subpopulation.append(self.create_rule(0.1, 1, 0.01))
        subpopulation.append(self.create_rule(0.2, 1, 0.01))
        subpopulation.append(self.create_rule(0.3, 1, 0.01))
        subpopulation.append(self.create_rule(0.4, 1, 0.01))
        subpopulation.append(self.create_rule(0.5, 1, 0.01))
        subpopulation.append(self.create_rule(0.6, 1, 0.01))
        subpopulation.append(self.create_rule(0.7, 1, 0.01))

        for i in range(50):
            mixing = mixing_model.NRandom(i, rule_amount)
            result = mixing.filter_subpopulation(subpopulation)
            print(result[0].fitness_)

        print("--------------------")
        mixing = mixing_model.NBestFitness(i, rule_amount)
        result = mixing.filter_subpopulation(subpopulation)
        print(result[0].fitness_)
        print(result[1].fitness_)
        print(result[2].fitness_)
        print(result[3].fitness_)

        print("--------------------")
        for i in range(50):
            mixing = mixing_model.RouletteWheel(i, rule_amount)
            result = mixing.filter_subpopulation(subpopulation)
            print(result[0].fitness_)
            print(result[1].fitness_)
            print(result[2].fitness_)
            print(result[3].fitness_)
            print("--------")
