# from suprb2.perf_recorder import PerfRecorder
from suprb2.config import Config
from suprb2.random_gen import Random
from suprb2.pool import ClassifierPool
from suprb2.classifier import Classifier
from sklearn.linear_model import LinearRegression

import numpy as np  # type: ignore
from copy import deepcopy
from abc import *

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
                                        Config().rule_discovery['mu'], False)
        for x in X[idxs]:
            cl = Classifier.random_cl(x)
            cl.fit(X, y)
            ClassifierPool().classifiers.append(cl)

        for i in Config().rule_discovery['nrules']:
            self.step(X, y)


class ES_MuLambd(RuleDiscoverer):
    def __init__(self):
        pass


    def step(self, X: np.ndarray, y: np.ndarray):
        for i in range(Config().rule_discovery['steps_per_step']):
            parents = self.select_parents_from_pool()
            recombined_classifiers = self.recombine(parents)
            children = self.mutate_and_fit(recombined_classifiers, X, y)
            next_generation = self.replace(parents, children)
            ClassifierPool().classifiers = next_generation


    def select_parents_from_pool(self):
        pool = ClassifierPool().classifiers
        mu = min(Config().rule_discovery['mu'], len(pool))
        parents = Random().random.choice(pool, mu, False)
        return parents


    def recombine(self, parents: np.ndarray):
        lmbd = Config().rule_discovery['lmbd']
        rho = Config().rule_discovery['rho']
        recombination_type = Config().rule_discovery['recombination']

        if recombination_type == 'intermediate':
            return self.intermediate_recombination(parents, lmbd, rho)
        elif recombination_type == 'discrete':
            return self.discrete_recombination(parents, lmbd, rho)
        else:
            return np.array([deepcopy(Random().random.choice(parents))])


    def intermediate_recombination(self, parents: np.ndarray, lmbd: int, rho: int):
        children = []
        for i in range(lmbd):
            candidates = Random().random.choice(parents, rho, False)
            averages = np.mean([[p.lowerBounds, p.upperBounds] for p in candidates], axis=0)
            children.append(Classifier(averages[0], averages[1],
                                            LinearRegression(), 1))  # Klaus: Change later
        return np.array(children)


    def discrete_recombination(self, parents: np.ndarray, lmbd: int, rho: int):
        children = []
        Xdim = parents[0].lowerBounds.size if type(parents[0].lowerBounds) == np.ndarray else 1
        for i in range(lmbd):
            candidates = Random().random.choice(parents, rho, False)
            lowerBounds = np.empty(Xdim)
            upperBounds = np.empty(Xdim)
            for i_dim in range(Xdim):
                lower = Random().random.choice([ c.lowerBounds[i_dim] for c in candidates ])
                upper = Random().random.choice([ c.upperBounds[i_dim] for c in candidates ])
                if lower > upper:
                    lowerBounds[i_dim] = upper
                    upperBounds[i_dim] = lower
                else:
                    lowerBounds[i_dim] = lower
                    upperBounds[i_dim] = upper
            children.append(Classifier(lowerBounds, upperBounds, LinearRegression(), 1))
        return np.array(children)



    def mutate_and_fit(self, classifiers: np.ndarray, X: np.ndarray, y:np.ndarray):
        children = []
        for cl in classifiers:
            cl.mutate(Config().rule_discovery['sigma'])
            cl.fit(X, y)
            default_error = self.default_error(y[np.nonzero(cl.matches(X))])
            if cl.error is not None and cl.error <= default_error:
                children.append(cl)
        return np.array(children)


    def replace(self, parents: np.ndarray, children: np.ndarray):
        next_generation = children
        if Config().rule_discovery['replacement'] == '+':
            next_generation = np.concatenate((children, parents))
        return next_generation


    @staticmethod
    def default_error(y: np.ndarray):
        if y.size == 0:
            return 0
        else:
            # for standardised data this should be equivalent to np.var(y)
            return np.sum(y**2)/len(y)
