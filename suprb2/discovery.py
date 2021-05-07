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
            cl = Classifier.random_cl(x, X.shape[1])
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

    'sigma'         ->  Positive scale factor for the mutation on the
                        classifiers' mutation vector (cl.sigmas).

    'replacement'   ->  This hyper parameter defines if we are also adding
                        the copies of the parents to the pool ('+'), or if
                        we are only adding the children (',').
    """


    def __init__(self):
        self.sigmas_dict = {}


    def step(self, X: np.ndarray, y: np.ndarray):
        generation = []

        # create start points for evolutionary search
        idxs = Random().random.choice(np.arange(len(X)),
								  Config().rule_discovery['mu'], False)
        for x in X[idxs]:
            cl = Classifier.random_cl(x, X.shape[1])
            cl.fit(X, y)
            generation.append(cl)

        # steps forward in the evolutionary search
        for i in range(Config().rule_discovery['steps_per_step']):
            recombined_classifiers = self.recombine(generation)
            children = self.mutate_and_fit(recombined_classifiers, X, y)
            generation = self.replace(generation, children)

        # add search results to pool
        mask = np.array([cl.get_weighted_error() < Utilities.default_error(y[np.nonzero(cl.matches(X))]) for cl in generation], dtype='bool')
        ClassifierPool().classifiers.extend(np.array(generation, dtype='object')[mask])


    def recombine(self, parents: List[Classifier]):
        if len(parents) == 0:
            return []

        lmbd = Config().rule_discovery['lmbd']
        rho = Config().rule_discovery['rho']
        recombination_type = Config().rule_discovery['recombination']

        if recombination_type == 'intermediate':
            return self.intermediate_recombination(parents, lmbd, rho)
        elif recombination_type == 'discrete':
            return self.discrete_recombination(parents, lmbd, rho)
        else:
            return [deepcopy(Random().random.choice(parents))]


    def intermediate_recombination(self, parents: List[Classifier], lmbd: int, rho: int):
        children = []
        x_dim = parents[0].lowerBounds.size if type(parents[0].lowerBounds) == np.ndarray else 1
        for i in range(lmbd):
            candidates = Random().random.choice(parents, rho, False)
            classifier_attrs = self.extract_classifier_attributes(candidates, rho, x_dim)

            avg_attrs = np.array([[np.mean(classifier_attrs[0].flatten())],
                                  [np.mean(classifier_attrs[1].flatten())],
                                  [np.mean(classifier_attrs[2].flatten())]])

            classifier = Classifier(avg_attrs[0], avg_attrs[1], LinearRegression(), 1) # Reminder: LinearRegression might change in the future
            self.get_classifier_sigmas(classifier, sigmas=avg_attrs[2])
            children.append(classifier)

        return children


    def discrete_recombination(self, parents: np.ndarray, lmbd: int, rho: int):
        children = []
        x_dim = parents[0].lowerBounds.size if type(parents[0].lowerBounds) == np.ndarray else 1
        for i in range(lmbd):
            candidates = Random().random.choice(parents, rho, False)
            classifier_attrs = self.extract_classifier_attributes(candidates, rho, x_dim)
            # select 'x_dim' values for each attribute (values are not crossed)
            selected_attr = np.array((Random().random.choice(classifier_attrs[0].flatten(), size=x_dim),
                                      Random().random.choice(classifier_attrs[1].flatten(), size=x_dim),
                                      Random().random.choice(classifier_attrs[2].flatten(), size=x_dim)), dtype=float)
            # flip if boundaries are inverted
            bounds = np.delete(selected_attr, -1, axis=0)
            sidx = bounds[:2].argsort(axis=0)
            bounds = bounds[sidx, np.arange(sidx.shape[1])]

            # create new classifier, and register
            classifier = Classifier(bounds[0], bounds[1], LinearRegression(), 1)
            self.get_classifier_sigmas(classifier, sigmas=selected_attr[2])

            children.append(classifier)
        return np.array(children)


    def extract_classifier_attributes(self, classifiers: List[Classifier], rho: int, x_dim: int):
        """
        Creates an array with shape (3, rho, x_dim),
        where 3 is the number of relevant attributes (lowers, uppers and sigmas).
        """
        classifier_attrs = np.zeros((3, rho, x_dim), dtype=float)
        for i in range(rho):
            cl = classifiers[i]
            classifier_attrs[0][i] = cl.lowerBounds
            classifier_attrs[1][i] = cl.upperBounds
            classifier_attrs[2][i] = self.get_classifier_sigmas(cl)
        return classifier_attrs


    def mutate_and_fit(self, classifiers: List[Classifier], X: np.ndarray, y:np.ndarray):
        """
        This method uses the traditional self-adapting ES mutation algortihm
        to mutate the classifiers.

        First, the classifier's mutation vector undergoes a mutation itself,
        influenced by two hyper parameters:
            'local_tau':  -> Scale used to calculate the local learning rate
                             (default value: 1.1)
            'global_tau': -> Scale used to calculate the global learning rate
                             (default value: 1.2)
        With the global and local learning rates calculated, the sigmas can be
        modified using the following formula:
            mutated_sigma = classifier_sigma * exp(N(1, tau_local) + N(1, tau_global))

        Each interval [l, u)'s bound x is changed to x' ~ N(x, mutated_sigmas(x)).
        A Gaussian distribution using values from the classifier's mutation
        vector as standard deviation.
        """
        children = []
        global_learning_rate = Random().random.normal(loc=1.0, scale=Config().rule_discovery['global_tau'])

        for cl in classifiers:
            # Apply global and local learning factors to the classifier's mutation vector
            sigmas = self.get_classifier_sigmas(cl)
            local_learning_rate = Random().random.normal(loc=1.0, scale=Config().rule_discovery['local_tau'])
            mutated_sigmas = sigmas * np.exp(local_learning_rate + global_learning_rate)

            # Mutate classifier's matching function
            lowers = Random().random.normal(loc=cl.lowerBounds, scale=mutated_sigmas, size=len(cl.lowerBounds))
            uppers = Random().random.normal(loc=cl.upperBounds, scale=mutated_sigmas, size=len(cl.upperBounds))
            lu = np.clip(np.sort(np.stack((lowers, uppers)), axis=0), a_max=1, a_min=-1)
            cl.lowerBounds = lu[0]
            cl.upperBounds = lu[1]
            # cl.mutate(Config().rule_discovery['sigma'])
            cl.fit(X, y)
            if cl.get_weighted_error() < Utilities.default_error(y[np.nonzero(cl.matches(X))]):
                children.append(cl)
        return children


    def get_classifier_sigmas(self, cl: Classifier, sigmas: np.ndarray=None):
        """
        Returns the mutation vector associated with this classifier,
        or it creates a new one.

        If sigmas i
        """
        cl_id = str(id(cl))
        if sigmas is not None:
            self.sigmas_dict.update(dict([(cl_id, sigmas)]))
        elif cl_id not in self.sigmas_dict:
            self.sigmas_dict.update({ cl_id: np.array(Random().random.normal(loc=1, scale=0.3, size=len(cl.lowerBounds)), dtype=float) })
        return self.sigmas_dict[cl_id]



    def replace(self, parents: List[Classifier], children: List[Classifier]):
        next_generation = children
        if Config().rule_discovery['replacement'] == '+':
            next_generation = children + parents
        return next_generation
