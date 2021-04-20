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


    def step(self, X: np.ndarray, y: np.ndarray):
        lmbd = Config().rule_discovery['lmbd']

        for i in range(Config().rule_discovery['steps_per_step']):
            children = np.array([])
            parents = self.remove_parents_from_pool()

            for j in range(lmbd):
                child = self.recombine(parents)
                child.mutate(Config().rule_discovery['sigma'])
                child.fit(X, y)
                children = np.append(children, child)

            classifiers = self.replace(parents, children)
            candidates = self.filter_classifiers(classifiers, X, y)
            next_generation = self.best_lambda_classifiers(candidates, lmbd)
            ClassifierPool().classifiers = next_generation


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


    def remove_parents_from_pool(self):
        pool = ClassifierPool().classifiers
        mu = min(Config().rule_discovery['mu'], len(pool))
        parents = Random().random.choice(pool, mu, False)

        ClassifierPool().classifiers = list(filter(lambda cl: cl not in parents, pool))
        return parents


    def recombine(self, parents: np.ndarray):
        if Config().rule_discovery['recombination'] == 'intermediate':
            averages = np.mean([[p.lowerBounds, p.upperBounds] for p in parents], axis=0)
            # Klaus: Only worried about the Linear Regression for now
            return Classifier(averages[0], averages[1], LinearRegression(), 1)
        else:
            return deepcopy(Random().random.choice(parents))


    def replace(self, parents: np.ndarray, children: np.ndarray):
        next_generation = children
        if Config().rule_discovery['replacement'] == '+':
            next_generation = np.concatenate((children, parents))
        return next_generation


    def filter_classifiers(self, classifiers: np.ndarray, X: np.ndarray, y: np.ndarray):
        filter_array = []
        for cl in classifiers:
            default_error = self.default_error(y[np.nonzero(cl.matches(X))])
            if cl.error is None or cl.error > default_error:
                filter_array.append(False)
            else:
                filter_array.append(True)

        return classifiers[filter_array]


    def best_lambda_classifiers(self, candidates: np.ndarray, lmbd: int):
        if candidates.size < lmbd:
            lmbd = candidates.size

        sorted_candidates = sorted(candidates, key=lambda cl: cl.error if cl.error is not None else float('inf'))
        return np.array(sorted_candidates[:lmbd])


    @staticmethod
    def default_error(y: np.ndarray):
        if y.size == 0:
            return 0
        else:
            # for standardised data this should be equivalent to np.var(y)
            return np.sum(y**2)/len(y)
