# from suprb2.perf_recorder import PerfRecorder
from suprb2.random_gen import Random
from suprb2.config import Config
from suprb2.classifier import Classifier
from suprb2.pool import ClassifierPool

import numpy as np  # type: ignore
from copy import deepcopy


class RuleDiscoverer:
    def __init__(self):
        pass

    def step(self, X, y):
        lmbd = Config().rule_discovery['lmbd']

        for i in range(Config().rule_discovery['steps_per_step']):
            children = np.array([])
            parents = self.take_parents_from_pool()

            for j in range(lmbd):
                child = deepcopy(Random().random.choice(parents)) # Klaus: Recombination operator!
                child.mutate(Config().rule_discovery['sigma'])
                child.fit(X, y)
                children = np.append(children, child)

            just_children = Config().rule_discovery['selection'] == ','
            next_generation = children if just_children else np.concatenate((children, parents))
            sorted_generation = sorted(next_generation, key=lambda cl: cl.error if cl.error is not None else float('inf'))
            ClassifierPool().classifiers = (self.filter_classifiers_by_error(sorted_generation, lmbd, X, y) + ClassifierPool().classifiers)


    def discover_rules(self, X, y):
        # draw n examples from data
        idxs = Random().random.choice(np.arange(len(X)),
                                        Config().rule_discovery['mu'], False)
        for x in X[idxs]:
            cl = Classifier.random_cl(x)
            cl.fit(X, y)
            ClassifierPool().classifiers.append(cl)

        for i in Config().rule_discovery['nrules']:
            self.step(X, y)

    def filter_classifiers_by_error(self, classifiers, lmbd, X, y):
        return [cl for cl in classifiers[:lmbd]
            if cl.error < self.default_error(y[np.nonzero(cl.matches(X))])]

    def take_parents_from_pool(self):
        parents = Random().random.choice(ClassifierPool().classifiers,
                                                Config().rule_discovery['mu'], False)
        ClassifierPool().classifiers = [cl for cl in ClassifierPool().classifiers if cl not in parents]
        return parents

    @staticmethod
    def default_error(y):
        if y.size == 0:
            return 0
        else:
            # for standardised data this should be equivalent to np.var(y)
            return np.sum(y**2)/len(y)
