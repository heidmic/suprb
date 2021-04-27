# from suprb2.perf_recorder import PerfRecorder
from suprb2.config import Config
from suprb2.random_gen import Random
from suprb2.pool import ClassifierPool
from suprb2.utilities import Utilities
from suprb2.classifier import Classifier
from sklearn.linear_model import LinearRegression

import numpy as np  # type: ignore
from copy import deepcopy
from abc import *
from typing import List

class RuleDiscoverer(ABC):
    def __init__(self):
        pass


    def step(self, X: np.ndarray, y: np.ndarray):
        pass


class ES_OnePlusLambd(RuleDiscoverer):
    def __init__(self):
        pass


    def step(self, X: np.ndarray, y: np.ndarray):
        # draw n examples from data
        idxs = Random().random.choice(np.arange(len(X)),
                                      Config().rule_discovery['nrules'], False)

        for x in X[idxs]:
            cl = Classifier.random_cl(x)
            cl.fit(X, y)
            for i in range(Config().rule_discovery['steps_per_step']):
                children = list()
                for j in range(Config().rule_discovery['lmbd']):
                    child = deepcopy(cl)
                    child.mutate(Config().rule_discovery['sigma'])
                    child.fit(X, y)
                    children.append(child)
                # ToDo instead of greedily taking the minimum, treating all
                #  below a certain threshhold as equal might yield better models
                cl = children[np.argmin([child.get_weighted_error() for child in children])]

            if cl.error < self.default_error(y[np.nonzero(cl.matches(X))]):
                ClassifierPool().classifiers.append(cl)


class ES_MuLambd(RuleDiscoverer):
    """
    This class represents the optimizer for rule generation.

    It uses Evolutionary Strategy generate an initial population
    with focus on diversity.

    Relevant hyper parameters for this class:
    'mu'            ->  Defines how many classifiers will be copied from the
                        pool for recombination.

    'lmbd'          ->  Number of children generated in the recombination.

    'rho'           ->  Indicates how many parents (out of 'mu') will be
                        used to generate one child.

    'steps_per_step'->  How many times we are going to repeat
                        the evolutionary search , when step()
                        is called. For instance, if steps_per_step
                        is 2, then we will select 'mu' classifiers
                        from the pool, recombine them into 'lmbd'
                        classifiers (and mutate them), 2 times.

    'recombination' ->  Indicates what kind of recombination will
                        be used to generate the children. The acceptables
                        values are:
                        'intermediate': lower and upper boundaries are
                        calculated from the average of 'rho' parents.
                        'discrete': lower and upper boundaries are
                        taken randomly from one of the 'rho' parents
                        for each Xdim.

    'mutation'      ->  Defines which mutation will be used on the
                        children. Acceptable values are:
                        'isotropic': lower and upper boundaries are
                        slightly deformed using a gaussian distribution.

    'sigma'         ->  Represents the size of the step we are taking
                        to a direction in the mutation process. In other
                        words, it shows us how big is the pertubation.

    'replacement'   ->  This hyper parameter defines if we are also adding
                        the copies of the parents to the pool ('+'), or if
                        we are only adding the children (',').
    """


    def __init__(self):
        pass


    def step(self, X: np.ndarray, y: np.ndarray):
        generation = []
        mu = Config().rule_discovery['mu']
        idxs = Random().random.choice(np.arange(len(X)), mu, False)

        for x in X[idxs]:
            cl = Classifier.random_cl(x)
            cl.fit(X, y)
            ClassifierPool().classifiers.append(cl)

        # evolutionary search
        for i in range(Config().rule_discovery['steps_per_step']):
            parents = deepcopy(Random().random.choice(ClassifierPool().classifiers, mu, False))
            recombined_classifiers = self.recombine(parents)
            children = self.mutate_and_fit(recombined_classifiers, X, y)
            generation.extend(self.replace(parents, children))

        # add search results to pool
        mask = [cl.get_weighted_error() < Utilities.default_error(y[np.nonzero(cl.matches(X))]) for cl in generation]
        if len(mask) > 0:
            ClassifierPool().classifiers.extend(generation[mask])


    def recombine(self, parents: List[Classifier]):
        if len(parents) == 0:
            return []

        lmbd = Config().rule_discovery['lmbd']
        if Config().rule_discovery['recombination'] == 'intermediate':
            return self.intermediate_recombination(parents, lmbd)
        else:
            return [deepcopy(Random().random.choice(parents))]


    def intermediate_recombination(self, parents: List[Classifier], lmbd: int):
        children = []
        rho = Config().rule_discovery['rho']
        for i in range(lmbd):
            candidates = Random().random.choice(parents, rho, False)
            averages = np.mean([[p.lowerBounds, p.upperBounds] for p in candidates], axis=0)
            children.append(Classifier(averages[0], averages[1],
                                            LinearRegression(), 1))  # Reminder: LinearRegression might change in the future
        return children


    def mutate_and_fit(self, classifiers: List[Classifier], X: np.ndarray, y:np.ndarray):
        children = []
        for cl in classifiers:
            cl.mutate(Config().rule_discovery['sigma'])
            cl.fit(X, y)
            if cl.get_weighted_error() < Utilities.default_error(y[np.nonzero(cl.matches(X))]):
                children.append(cl)
        return children


    def replace(self, parents: List[Classifier], children: List[Classifier]):
        next_generation = children
        if Config().rule_discovery['replacement'] == '+':
            next_generation = children + parents
        return next_generation
