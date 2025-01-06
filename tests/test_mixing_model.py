import unittest
import numpy as np

from suprb import SupRB
from suprb.utils import check_random_state
from suprb.rule import Rule
from suprb.rule.fitness import VolumeWu
from suprb.rule.matching import OrderedBound

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
            subpopulation.append(self.create_rule(i/100, i, 0.01))

        random_state.shuffle(subpopulation)

        return subpopulation

    def setUp(self):
        self.subpopulation = self.create_subpopulation()

    def test_NBestFitness(self):
        rule_amount = 4
        random_state = 42

        mixing = mixing_model.NBestFitness(rule_amount, random_state)
        result = mixing(self.subpopulation)

        self.assertEqual(len(result), 4)
        self.assertEqual(result[0].fitness_, 0.96)
        self.assertEqual(result[1].fitness_, 0.97)
        self.assertEqual(result[2].fitness_, 0.98)
        self.assertEqual(result[3].fitness_, 0.99)

    def test_NRandom(self):
        rule_amount = 4
        random_state = 42

        # Elements are not checked because they differ for different python versions

        mixing = mixing_model.NRandom(rule_amount, random_state)
        result = mixing(self.subpopulation)

        self.assertEqual(len(result), 4)

        random_state = 1
        mixing = mixing_model.NRandom(rule_amount, random_state)
        result = mixing(self.subpopulation)

        self.assertEqual(len(result), 4)

    def test_RouletteWheel(self):
        rule_amount = 4
        random_state = 42

        mixing = mixing_model.RouletteWheel(rule_amount, random_state)
        result = mixing(self.subpopulation)

        self.assertEqual(len(result), 4)
        self.assertEqual(result[0].fitness_, 0.74)
        self.assertEqual(result[1].fitness_, 0.68)
        self.assertEqual(result[2].fitness_, 0.90)
        self.assertEqual(result[3].fitness_, 0.72)

        random_state = 1
        mixing = mixing_model.RouletteWheel(rule_amount, random_state)
        result = mixing(self.subpopulation)

        self.assertEqual(len(result), 4)
        self.assertEqual(result[0].fitness_, 0.93)
        self.assertEqual(result[1].fitness_, 0.66)
        self.assertEqual(result[2].fitness_, 0.99)
        self.assertEqual(result[3].fitness_, 0.13)

    def test_ExperienceCalculation(self):
        experience_calc = mixing_model.ExperienceCalculation()
        result_experiences = sorted(experience_calc(self.subpopulation))

        for i in range(100):
            self.assertEqual(result_experiences[i], i)

    def test_CapExperience(self):
        for i in range(100):
            experience_calc = mixing_model.CapExperience(i/2, i)
            result_experiences = experience_calc(self.subpopulation)
            self.assertEqual(max(result_experiences), i)
            self.assertEqual(min(result_experiences), i/2)

    def test_CapExperienceWithDimensionality(self):
        lower = 5
        upper = 20
        experience_calc = mixing_model.CapExperienceWithDimensionality(
            lower, upper)

        result_experiences = experience_calc(self.subpopulation, 2)
        self.assertEqual(max(result_experiences), 40)
        self.assertEqual(min(result_experiences), 10)

        result_experiences = experience_calc(self.subpopulation, 4)
        self.assertEqual(max(result_experiences), 80)
        self.assertEqual(min(result_experiences), 20)

        result_experiences = experience_calc(self.subpopulation, 6)
        self.assertEqual(max(result_experiences), 99)
        self.assertEqual(min(result_experiences), 30)
