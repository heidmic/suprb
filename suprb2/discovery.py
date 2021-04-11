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

    def step(self, X_val, y_val):
        pass

    def discover_rules(self, X, y):
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
            cl = children[np.argmin([child.error for child in children])]
        if cl.error < self.default_error(y[np.nonzero(cl.matches(X))]):
            ClassifierPool().classifiers.append(cl)

    @staticmethod
    def default_error(y):
        # for standardised data this should be equivalent to np.var(y)
        return np.sum(y**2)/len(y)
