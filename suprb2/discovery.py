# from suprb2.perf_recorder import PerfRecorder
from suprb2.config import Config
from suprb2.random_gen import Random
from suprb2.utilities import Utilities
from suprb2.classifier import Classifier
from sklearn.linear_model import LinearRegression

import numpy as np  # type: ignore
from copy import deepcopy
from abc import *
from typing import List, Tuple

class RuleDiscoverer(ABC):
    def __init__(self, pool: List[Classifier]) -> None:
        self.pool = pool


    def step(self, X: np.ndarray, y: np.ndarray) -> None:
        pass


    def extract_classifier_attributes(self, classifiers_tuples: List[Tuple[Classifier, np.ndarray]], x_dim: int, rho: int=None) -> np.ndarray:
        """
        Creates an array with shape (3, rho, x_dim),
        where 3 is the number of relevant attributes (lowers, uppers and sigmas).
        """
        rho = rho if rho is not None else len(classifiers_tuples)
        classifier_attrs = np.zeros((3, rho, x_dim), dtype=float)
        for i in range(rho):
            classifier_attrs[:,i] = np.array([  classifiers_tuples[i][0].lowerBounds,
                                                classifiers_tuples[i][0].upperBounds,
                                                classifiers_tuples[i][1] ],
                                            dtype=object).reshape((3, 1))

        return classifier_attrs


    def create_sigmas(self, x_dim: int) -> np.ndarray:
        """
        Creates an array with size 'x_dim' with
        uniformly distributed values from [0, 1]
        """
        return Random().random.uniform(size=x_dim)


class ES_OnePlusLambd(RuleDiscoverer):
    def __init__(self, pool: List[Classifier]):
        super().__init__(pool)


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
                self.pool.append(cl)


class ES_MuLambd(RuleDiscoverer):
    """
    This class represents the optimizer for rule generation.

    It uses Evolution Strategies generate an initial population
    with focus on diversity.

    Its pool is a list of tuple, where each tuple
    has an classifier and a mutation vector (sigmas)
    associated with that classifier.

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

    'local_tau'     ->  Scale used to calculate the local learning rate
                        for the classifier's mutation.

    'global_tau'    ->  Scale used to calculate the global learning rate
                        for the classifier's mutation.

    'replacement'   ->  This hyper parameter defines if we are also adding
                        the copies of the parents to the pool ('+'), or if
                        we are only adding the children (',').
    """


    def __init__(self, pool: List[Tuple[Classifier, np.ndarray]]) -> None:
        super().__init__(pool)
        self.sigmas_dict = {}


    def step(self, X: np.ndarray, y: np.ndarray) -> None:
        generation_tuples = list()
        mu = Config().rule_discovery['mu']
        x_dim = X.shape[1]

        # create start points for evolutionary search (with mutation vectors)
        idxs = Random().random.choice(np.arange(len(X)), mu, False)
        for x in X[idxs]:
            cl = Classifier.random_cl(xdim=x_dim, point=x)
            cl.fit(X, y)
            generation_tuples.append( (cl, self.create_sigmas(x_dim)) )

        # steps forward in the evolutionary search
        for i in range(Config().rule_discovery['steps_per_step']):
            recmb_tuples      = self.recombine(generation_tuples)
            children_tuples   = self.mutate_and_fit(recmb_tuples, X, y)
            generation_tuples = self.replace(generation_tuples, children_tuples)

        # add search results to pool
        mask = np.array([cl_tuple[0].get_weighted_error() < Utilities.default_error(y[np.nonzero(cl_tuple[0].matches(X))]) for cl_tuple in generation_tuples], dtype='bool')
        self.pool.extend( list(np.array(generation_tuples, dtype=object)[mask]) )


    def recombine(self, parents_tuples: List[Tuple[Classifier, np.ndarray]]) -> List[Tuple[Classifier, np.ndarray]]:
        """
        This method decides which kind of recombination will be done,
        according to the rule discovery hyper parameter: 'recombination'
        If 'recombination' == 'intermediate', then new 'lmbd' classifiers
        will be created out of the mean of 'rho' parents attributes.
        If 'recombination' == 'discrete', then the new classifiers are
        created randomly from the attributes from the parents
        (attributes do not cross values).
        If 'recombination' is somethin else, then only one classifier
        will be created and it will be a copy from one of the parents.
        """
        if len(parents_tuples) == 0:
            return None

        lmbd = Config().rule_discovery['lmbd']
        rho = Config().rule_discovery['rho']
        recombination_type = Config().rule_discovery['recombination']

        if recombination_type == 'intermediate':
            return self.intermediate_recombination(parents_tuples, lmbd, rho)
        elif recombination_type == 'discrete':
            return self.discrete_recombination(parents_tuples, lmbd, rho)
        else:
            cl_index = Random().random.choice(range(len(parents_tuples)))
            copied_tuple = deepcopy(parents_tuples[cl_index])
            return copied_tuple


    def intermediate_recombination(self, parents_tuples: List[Tuple[Classifier, np.ndarray]], lmbd: int, rho: int) -> List[Tuple[Classifier, np.ndarray]]:
        """
        This methods creates 'lmbd' classifiers from the average
        of 'rho' classifiers in parents.
        So, the new classifier look like this:
        Classifier( lowers=average(rho_candidates.lowerBounds),
                    uppers=average(rho_candidates.upperBounds),
                    LinearRegression(), degree=1 )
        Classifier's sigmas = average(rho_candidates.sigmas)
        """
        children_tuples = list()
        parents_array = np.array(parents_tuples, dtype=object)
        x_dim = parents_array[0,0].lowerBounds.size if type(parents_array[0,0].lowerBounds) == np.ndarray else 1
        for i in range(lmbd):
            candidates_tuples = Random().random.choice(parents_array, rho, False)
            classifier_attrs = self.extract_classifier_attributes(candidates_tuples, x_dim=x_dim, rho=rho)

            avg_attrs = np.array([[np.mean(classifier_attrs[0].flatten())],
                                  [np.mean(classifier_attrs[1].flatten())],
                                  [np.mean(classifier_attrs[2].flatten())]])

            classifier = Classifier(avg_attrs[0], avg_attrs[1], LinearRegression(), 1) # Reminder: LinearRegression might change in the future
            children_tuples.append((classifier, avg_attrs[2]))

        return children_tuples


    def discrete_recombination(self, parents_tuples: List[Tuple[Classifier, np.ndarray]], lmbd: int, rho: int) -> List[Tuple[Classifier, np.ndarray]]:
        """
        This method creates 'lmbd' classifiers picking randomly
        the attributes from 'rho' parents. The values do not cross
        types (so, values used for upperBounds, can only be used for
        one classifier's upperBounds).
        If the values are flipped (lowerBounds > upperBounds), then
        unflip it and save.
        """
        children_tuples = list()
        parents_array = np.array(parents_tuples, dtype=object)
        x_dim = parents_array[0,0].lowerBounds.size if type(parents_array[0,0].lowerBounds) == np.ndarray else 1
        for i in range(lmbd):
            candidates_tuples = Random().random.choice(parents_array, rho, False)
            classifier_attrs = self.extract_classifier_attributes(candidates_tuples, x_dim=x_dim, rho=rho)
            # select 'x_dim' values for each attribute (values are not crossed)
            selected_attr = np.array((Random().random.choice(classifier_attrs[0].flatten(), size=x_dim),
                                      Random().random.choice(classifier_attrs[1].flatten(), size=x_dim),
                                      Random().random.choice(classifier_attrs[2].flatten(), size=x_dim)), dtype=float)
            # flip if boundaries are inverted
            bounds = np.delete(selected_attr, -1, axis=0)
            sidx = bounds[:2].argsort(axis=0)
            bounds = bounds[sidx, np.arange(sidx.shape[1])]

            # create new classifier, and save sigmas
            classifier = Classifier(bounds[0], bounds[1], LinearRegression(), 1)
            children_tuples.append( (classifier, selected_attr[2]) )

        return children_tuples


    def mutate_and_fit(self, cls_tuples: List[Tuple[Classifier, np.ndarray]], X: np.ndarray, y:np.ndarray) -> List[Tuple[Classifier, np.ndarray]]:
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
        children_tuples = list()
        global_learning_rate = Random().random.normal(scale=Config().rule_discovery['global_tau'])
        if cls_tuples is None:
            return children_tuples

        cls_len = len(cls_tuples)

        for i in range(cls_len):
            cl, cl_sigmas = cls_tuples[i]

            # Apply global and local learning factors to the classifier's mutation vector
            local_learning_rate = Config().rule_discovery['local_tau'] * Random().random.normal(size=X.shape[1])
            mutated_sigmas = cl_sigmas * np.exp(local_learning_rate + global_learning_rate)

            # Mutate classifier's matching function
            lowers = Random().random.normal(loc=cl.lowerBounds, scale=mutated_sigmas, size=X.shape[1])
            uppers = Random().random.normal(loc=cl.upperBounds, scale=mutated_sigmas, size=X.shape[1])
            lu = np.clip(np.sort(np.stack((lowers, uppers)), axis=0), a_max=1, a_min=-1)
            cl.lowerBounds = lu[0]
            cl.upperBounds = lu[1]
            # cl.mutate(Config().rule_discovery['sigma'])
            cl.fit(X, y)
            if cl.get_weighted_error() < Utilities.default_error(y[np.nonzero(cl.matches(X))]):
                children_tuples.append((cl, cl_sigmas))

        return children_tuples


    def replace(self, parents_tuples: List[Tuple[Classifier, np.ndarray]], children_tuples: List[Tuple[Classifier, np.ndarray]]) -> List[Tuple[Classifier, np.ndarray]]:
        """
        This method uses the hyper parameter to define if
        in the next generation, there will be only the
        children ('replacement' == ','), or if the parents
        will also be included ('replacement' == '+').
        Default value for 'replacement' is '+'.
        """
        if Config().rule_discovery['replacement'] == '+':
            return parents_tuples + children_tuples
        else:
            return children_tuples


class ES_MuLambdSearchPath(RuleDiscoverer):
    """
    This optimizer uses the Evolution Strategy
    Cumullative Step-Size Adaptation in order
    to promote diversity in the population.

    Its pool is a list of tuple, where each tuple
    has an classifier and a mutation vector (sigmas)
    associated with that classifier.

    Relevant hyper parameters are:
        'lmbd':             Number of new classifier generated.
                            After the 'lmbd' classifiers are
                            generated, only 'mu' will be selected
                            (according to the fitness/error).
                            Recommended value: positive int

        'mu':               Number of the best 'mu' new classifiers
                            that are going to be selected as parents
                            for the new classifier.
                            Recommended value: 'lmbd'/4

        'steps_per_step':   'steps_per_step'->  How many times we are going
                            to repeat the evolutionary search , when step()
                            is called. For instance, if steps_per_step
                            is 2, then run 2 steps in the evolutionary
                            search started by step().

    Implementation based on paper ES Overview 2015 by Hansen, Arnold & Auger.
    Page 13, Algorithm 4 - The (μ/μ, λ)-ES with Search Path)
    Links:
    - PDF download: https://hal.inria.fr/hal-01155533/file/es-overview-2015.pdf
    - Refence: https://scholar.google.com/citations?user=NsIbm80AAAAJ&hl=en#
    """


    def __init__(self, pool: List[Tuple[Classifier, np.ndarray]]) -> None:
        super().__init__(pool)


    def step(self, X: np.ndarray, y: np.ndarray) -> None:
        lmbd            = Config().rule_discovery['lmbd']
        mu              = Config().rule_discovery['mu']
        x_dim           = X.shape[1]
        sigma_coef      = np.sqrt(mu / (x_dim + mu))
        dist_global     = 1 + np.sqrt(mu / x_dim)
        dist_local      = 3 * x_dim
        start_point    = [Classifier.random_cl(x_dim), self.create_sigmas(x_dim)]
        tuples_for_pool = list()
        search_path     = 0

        for i in range(Config().rule_discovery['steps_per_step']):
            rnd_tuple_list = list()

            # generating children with sigmas
            for j in range(lmbd):
                cl = deepcopy(start_point[0])
                cl.fit(X, y)
                rnd_tuple_list.append( [cl, (start_point[1] * self.create_sigmas(x_dim))] )
            children_tuple_list = np.array(self.select_best_classifiers(rnd_tuple_list, mu))
            tuples_for_pool.extend( children_tuple_list )

            # recombination and parent update
            search_path = (1 - sigma_coef) * search_path + np.sqrt(sigma_coef * (2 - sigma_coef)) * (np.sqrt(mu) / mu) * np.sum(children_tuple_list[:,1])

            # Local changes:
            # Instead of using the expected value of a zero mean unit variance normal
            # distribution (Algorthm 4: line 8b), we are going to use the expected
            # value of the half normal distribution, which is equivalent to the previous
            # expeted value.
            # E[Y] = mu = 1 * sqrt(2) / sqrt(pi)
            local_expected_value = np.sqrt(2) / np.sqrt(np.pi)
            local_factor = np.power(( np.exp((np.abs(search_path) / local_expected_value) - 1) ),  (1 / dist_local))

            # Global changes:
            global_expected_value = np.linalg.norm(Random().random.normal(scale=np.identity(x_dim)))
            global_factor = np.power(( np.exp((np.linalg.norm(search_path) / global_expected_value) - 1) ), (sigma_coef / dist_global))

            # step-size changes
            start_point[1] = start_point[1] * local_factor * global_factor

            # recombining parents attributes
            parents_attr = self.extract_classifier_attributes(children_tuple_list, x_dim)
            start_point[0].lowerBounds = (1/mu) * np.sum(parents_attr[0])
            start_point[0].upperBounds = (1/mu) * np.sum(parents_attr[1])

        # add children to pool
        self.pool.extend( tuples_for_pool )


    def select_best_classifiers(self, tuple_list: List[Tuple[Classifier, np.ndarray]], mu: int) -> List[Tuple[Classifier, np.ndarray]]:
        """
        Return the 'mu' best classifiers (according to their weighted error)
        from the 'tuple_list'. If mu <= len(tuple_list), then ValueError is raised.
        """
        tuple_array = np.array(tuple_list, dtype=object)
        idx = np.argpartition([ cl_tuple[0].get_weighted_error() for cl_tuple in tuple_array ], mu).astype(int)[:mu]
        return list(tuple_array[idx])
