# from suprb2.perf_recorder import PerfRecorder
from suprb2.config import Config
from suprb2.random_gen import Random
from suprb2.pool import ClassifierPool
from suprb2.utilities import Utilities
from suprb2.classifier import Classifier
from suprb2.solutions import SolutionOptimizer
from sklearn.linear_model import LinearRegression

import numpy as np  # type: ignore
from copy import deepcopy
from abc import *
from typing import List

class RuleDiscoverer(ABC):
    def __init__(self):
        pass


    def step(self, X: np.ndarray, y: np.ndarray, solution_opt: SolutionOptimizer=None):
        raise NotImplementedError()


    def create_start_points(self, n: int, X: np.ndarray, y: np.ndarray, solution_opt: SolutionOptimizer=None):
        """
        This method creates classifier as starting points for
        an evolutionary search.
        There are 3 different strategies:
            - 'draw_examples_from_data'
            - 'elitist_unmatched'
            - 'elitist_compliment'
        """
        technique = Config().rule_discovery['start_points']

        if technique == 'elitist_compliment':
            return self.elitist_compliment(solution_opt)
        elif technique == 'elitist_unmatched':
            return self.elitist_unmatched(n, solution_opt)
        else:
            return self.draw_examples_from_data(n, X, y)


    def elitist_compliment(self, solution_opt: SolutionOptimizer):
        """
        This method takes the classifiers from the elitist Individual
        and extract the compliment of their matching intervals [l, u).
        after that, we distribute the intervals compliments throughout
        the starting point classifiers.
        """
        start_points = deepcopy(solution_opt.get_elitist().get_classifiers())
        for cl in start_points:
            pass


    def elitist_unmatched(self, n: int, solution_opt: SolutionOptimizer):
        """
        This method takes 'n' examples out of the inputs that were
        not matched by the elitist individual.
        """
        classifiers = solution_opt.get_elitist().get_classifiers()
        # start_points = classifiers.


    def draw_examples_from_data(self, n: int, X: np.ndarray, y: np.ndarray):
        """
        This method takes 'n' random examples out of the inputs and
        creates one classifier for each example taken.
        """
        start_points = []
        idxs = Random().random.choice(np.arange(len(X)), n, False)
        for x in X[idxs]:
            cl = Classifier.random_cl(x, X.shape[1])
            cl.fit(X, y)
            start_points.append(cl)
        return start_points


class ES_OnePlusLambd(RuleDiscoverer):
    def __init__(self):
        pass


    def step(self, X: np.ndarray, y: np.ndarray, solution_opt: SolutionOptimizer=None):
        mu = Config().rule_discovery['mu']

        for cl in self.create_start_points(mu, X, y, solution_opt):
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
        pass


    def step(self, X: np.ndarray, y: np.ndarray, solution_opt: SolutionOptimizer=None):
        # create start points for evolutionary search
        mu = Config().rule_discovery['mu']
        generation = self.create_start_points(mu, X, y, solution_opt)

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
        for i in range(lmbd):
            candidates = Random().random.choice(parents, rho, False)
            boundaries_avg = np.mean([[p.lowerBounds, p.upperBounds] for p in candidates], axis=0)
            sigmas_avg = np.mean([p.sigmas for p in candidates], axis=0)
            copy_sigmas = sigmas_avg.copy()
            copy_avg = boundaries_avg.copy()
            children.append(Classifier(copy_avg[0], copy_avg[1],
                                            LinearRegression(), 1, copy_sigmas))  # Reminder: LinearRegression might change in the future
        return children


    def discrete_recombination(self, parents: np.ndarray, lmbd: int, rho: int):
        children = []
        Xdim = parents[0].lowerBounds.size if type(parents[0].lowerBounds) == np.ndarray else 1
        for i in range(lmbd):
            candidates = Random().random.choice(parents, rho, False)
            lowerBounds = np.empty(Xdim)
            upperBounds = np.empty(Xdim)
            sigmas = np.empty(Xdim)

            for i_dim in range(Xdim):
                sigmas[i_dim] = Random().random.choice([ c.sigmas[i_dim] for c in candidates ])
                lower = Random().random.choice([ c.lowerBounds[i_dim] for c in candidates ])
                upper = Random().random.choice([ c.upperBounds[i_dim] for c in candidates ])
                # flip if boundaries are inverted
                lowerBounds[i_dim], upperBounds[i_dim] = sorted((lower, upper))

            children.append(Classifier(lowerBounds, upperBounds, LinearRegression(), 1, sigmas))
        return np.array(children)


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
