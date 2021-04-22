# from suprb2.perf_recorder import PerfRecorder
from suprb2.random_gen import Random
from suprb2.config import Config
from suprb2.classifier import Classifier
from suprb2.pool import ClassifierPool
from sklearn.linear_model import LinearRegression

import numpy as np  # type: ignore
from copy import deepcopy

class RuleDiscoverer:
    def __init__(self):
        pass


    def discover_initial_rules(self, X: np.ndarray, y: np.ndarray):
        # draw n examples from data
        idxs = Random().random.choice(np.arange(len(X)),
                                        Config().rule_discovery['mu'], False)
        for x in X[idxs]:
            cl = Classifier.random_cl(x)
            cl.fit(X, y)
            ClassifierPool().classifiers.append(cl)

        for i in Config().rule_discovery['nrules']:
            self.step(X, y)


    def step(self, X: np.ndarray, y: np.ndarray):
        for i in range(Config().rule_discovery['steps_per_step']):
            parents = self.remove_parents_from_pool()
            recombined_classifiers = self.recombine(parents)
            children = self.mutate_and_fit(recombined_classifiers, X, y)
            next_generation = self.replace(parents, children)
            ClassifierPool().classifiers = next_generation


    def remove_parents_from_pool(self):
        pool = ClassifierPool().classifiers
        mu = min(Config().rule_discovery['mu'], len(pool))
        parents = Random().random.choice(pool, mu, False)

        ClassifierPool().classifiers = list(filter(lambda cl: cl not in parents, pool))
        return parents


    def recombine(self, parents: np.ndarray):
            # next_generation = self.best_lambda_classifiers(candidates, lmbd)
        lmbd = Config().rule_discovery['lmbd']
        if Config().rule_discovery['recombination'] == 'intermediate':
            return self.intermediate_recombination(parents, lmbd)
        else:
            return np.array([deepcopy(Random().random.choice(parents))])


    def intermediate_recombination(self, parents: np.ndarray, lmbd: int):
        children = []
        for i in range(lmbd):
            couple = Random().random.choice(parents, 2, False)
            averages = np.mean([[p.lowerBounds, p.upperBounds] for p in couple], axis=0)
            children.append(Classifier(averages[0], averages[1],
                                            LinearRegression(), 1))  # Klaus: Change later
        return np.array(children)


    def mutate_and_fit(self, classifiers: np.ndarray, X: np.ndarray, y:np.ndarray):
        children = []
        for cl in classifiers:
            cl.mutate(Config().rule_discovery['sigma'])
            cl.fit(X, y)
            default_error = self.default_error(y[np.nonzero(cl.matches(X))])
            if cl.error is not None and cl.error < default_error:
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
