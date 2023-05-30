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


class TestMixingModel(unittest.TestCase):

    def create_rule(self, fitness, experience, error):
        rule = Rule(match=OrderedBound(np.array([[-1, 1]])),
                    input_space=[-1.0, 1.0],
                    model=SupRB(),
                    fitness=VolumeWu)

        rule.fitness_ = fitness
        rule.experience_ = experience
        rule.error_ = error

        return rule

    def create_subpopulation(self):
        subpopulation = []
        random_state = check_random_state(42)

        for i in range(100):
            subpopulation.append(self.create_rule(i/100, 1, 0.01))

        random_state.shuffle(subpopulation)

        return subpopulation

    def test_NBestFitness_filter_subpopulation(self):
        rule_amount = 4
        random_state = 42

        subpopulation = self.create_subpopulation()

        mixing = mixing_model.NBestFitness(rule_amount, random_state)
        result = mixing(subpopulation)

        self.assertEqual(len(result), 4)
        self.assertEqual(result[0].fitness_, 0.96)
        self.assertEqual(result[1].fitness_, 0.97)
        self.assertEqual(result[2].fitness_, 0.98)
        self.assertEqual(result[3].fitness_, 0.99)

    def test_NRandom_filter_subpopulation(self):
        rule_amount = 4
        random_state = 42

        subpopulation = self.create_subpopulation()

        mixing = mixing_model.NRandom(rule_amount, random_state)
        result = mixing(subpopulation)

        self.assertEqual(len(result), 4)
        self.assertEqual(result[0].fitness_, 0.92)
        self.assertEqual(result[1].fitness_, 0.88)
        self.assertEqual(result[2].fitness_, 0.09)
        self.assertEqual(result[3].fitness_, 0.05)

        random_state = 1
        mixing = mixing_model.NRandom(rule_amount, random_state)
        result = mixing(subpopulation)

        self.assertEqual(len(result), 4)
        self.assertEqual(result[0].fitness_, 0.84)
        self.assertEqual(result[1].fitness_, 0.93)
        self.assertEqual(result[2].fitness_, 0.58)
        self.assertEqual(result[3].fitness_, 0.66)

    def test_RouletteWheel_filter_subpopulation(self):
        rule_amount = 4
        random_state = 42

        subpopulation = self.create_subpopulation()

        mixing = mixing_model.RouletteWheel(rule_amount, random_state)
        result = mixing(subpopulation)

        self.assertEqual(len(result), 4)
        self.assertEqual(result[0].fitness_, 0.74)
        self.assertEqual(result[1].fitness_, 0.68)
        self.assertEqual(result[2].fitness_, 0.90)
        self.assertEqual(result[3].fitness_, 0.72)

        random_state = 1
        mixing = mixing_model.RouletteWheel(rule_amount, random_state)
        result = mixing(subpopulation)

        self.assertEqual(len(result), 4)
        self.assertEqual(result[0].fitness_, 0.93)
        self.assertEqual(result[1].fitness_, 0.66)
        self.assertEqual(result[2].fitness_, 0.99)
        self.assertEqual(result[3].fitness_, 0.13)
