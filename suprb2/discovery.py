from __future__ import annotations

import numpy as np  # type: ignore
from copy import deepcopy
from abc import *
from sklearn.linear_model import LinearRegression, LogisticRegression

# from suprb2.perf_recorder import PerfRecorder
from suprb2.solutions import SolutionOptimizer
from suprb2.classifier import Classifier
from suprb2.utilities import Utilities
from suprb2.random_gen import Random
from suprb2.config import Config

class RuleDiscoverer(ABC):
    """
    Abstract class from which all optimizers inherit from.
    Important hyperparameters:
        'name': (method 'select_best_classifiers') determs
                which optimizer to use. The reason for this is purely
                based on the automatization of experiments which
                different configurations.
        'start_point': (method 'create_start_tuples') determs
                        which strategy is used to create the initial
                        classifiers. Acceptable values are:
                        u -> (elitist) unmatched
                        c -> (elitist) complement
                        None or something else -> draw examples from data
    """


    def __init__(self, pool: list[Classifier], solution_optimizer: SolutionOptimizer=None) -> None:
        self.pool = pool
        self.sol_opt = solution_optimizer


    def step(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError()


    def extract_classifier_attributes(self, classifiers_tuples: list[tuple[Classifier, np.ndarray]], x_dim: int, rho: int=None) -> np.ndarray:
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
                                            dtype=object).reshape((3, x_dim))

        return classifier_attrs


    def create_sigmas(self, x_dim: int) -> np.ndarray:
        """
        Creates an array with size 'x_dim' with
        uniformly distributed values from [0, 1]
        """
        return Random().random.uniform(size=x_dim)


    def select_best_classifiers(self, tuple_list: list[tuple[Classifier, np.ndarray]], mu: int) -> list[tuple[Classifier, np.ndarray]]:
        """
        Return the 'mu' best classifiers (according to their weighted error)
        from the 'tuple_list'. If mu < len(tuple_list), then ValueError is raised.
        """
        if mu == len(tuple_list):
            return tuple_list
        else:
            tuple_array = np.array(tuple_list, dtype=object)
            idx = np.argpartition([ cl_tuple[0].get_weighted_error() for cl_tuple in tuple_array ], mu-1).astype(int)[:(mu)]
            return list(tuple_array[idx])


    def get_rule_disc(pool: list[Classifier]) -> RuleDiscoverer:
        """
        Returns the optimizer pointed by the rule discovery
        hyper-parameter 'name'.
        """
        optimizer = Config().rule_discovery['name']
        if optimizer == 'ES_OPL':
            return ES_OnePlusLambd(pool)
        elif optimizer == 'ES_ML':
            return ES_MuLambd(pool)
        elif optimizer == 'ES_MLSP':
            return ES_MuLambdSearchPath(pool)
        elif optimizer == 'ES_CMA':
            return ES_CMA(pool)
        else:
            raise NotImplemented


    def create_start_tuples(self, n: int, X: np.ndarray, y: np.ndarray) -> list(tuple(Classifier, np.ndarray)):
        """
        This method creates classifier as starting points for
        an evolutionary search.
        There are 3 different strategies:
            - 'draw_examples_from_data' (Config().rule_discovery['start_points'] = 'd')
            - 'elitist_complement'      (Config().rule_discovery['start_points'] = 'c')
            - 'elitist_unmatched'       (Config().rule_discovery['start_points'] = 'u')
        """
        technique = Config().rule_discovery['start_points']

        if technique == 'd' or self.sol_opt is None:
            return self.draw_examples_from_data(n, X, y)
        if technique == 'c':
            # second return value is the array of intervals (for test)
            classifiers, _ = self.elitist_complement(n, X, y)
            return classifiers
        elif technique == 'u':
            return self.elitist_unmatched(n, X, y)
        else:
            raise NotImplementedError


    def elitist_complement(self, n: int, X: np.ndarray, y: np.ndarray) -> list(tuple(Classifier, np.ndarray)):
        """
        This method takes the classifiers from the elitist Individual
        and extract the complement of their matching intervals [l, u).
        after that, we distribute 1/nth of the intervals' complement
        among the starting point classifiers.

        It calculates the complements/intervals in an array with
        shape = (cl_num, Xdim, n * 2, 2).
        'cl_num' is the number of classifiers matched by the elitist
        individual.
        'Xdim' is the dimension of the input; 'n' is the number of
        intervals created pro complement.
        The third dimension in the calculated array (n * 2) is the
        number of parts the complement will be sliced into.
        'n' is multiplied by 2, because each complement generates 2
        intervals ([-1, l] and [u, 1]) (one or both of theses intervals
        might be empty, for example: [-1, -1] and [0, 1]).
        The last dimension is 2, because of the interval representation,
        which is a two elements array.
        """
        start_tuples = list()
        elitist_classifiers = self.sol_opt.get_elitist().get_classifiers()
        cl_num, xdim = (len(elitist_classifiers), X.shape[1])
        intervals = np.zeros((cl_num, xdim, n * 2, 2), dtype=float)
        for i in range(cl_num):
            cl = elitist_classifiers[i]

            # Split all the complements in all xdim of the i-th classifier,
            # and save it in the i-th line of the array
            for j in range(xdim):
                lower = [ -1, cl.lowerBounds[j]]
                upper = [cl.upperBounds[j], 1]
                np.concatenate((self.split_interval(lower, n),
                                self.split_interval(upper, n)),
                                axis=0, out=intervals[i, j])

            # with the i-th line, create n * 2 classifiers
            for k in range(n * 2):
                new_classifier = Classifier(lowers=intervals[i,:,k,0], uppers=intervals[i,:,k,1], degree=1)
                new_classifier.fit(X, y)
                start_tuples.append( [new_classifier, self.create_sigmas(xdim)] )

        return (start_tuples, intervals)


    def split_interval(self, l: np.ndarray, n: int) -> np.ndarray:
        """
        This method splits an interval 'l' into 'n' new ones.
        For example:
            interval([10, 30], 2) => [[10, 20], [20, 30]]
            interval([10, 30], 4) => [[10, 15], [15, 20], [20, 25], [25, 30]]
        """
        w = (l[1] - l[0]) / n
        return np.array([ [l[0]+i*w, l[0]+(i+1)*w] for i in range(n) ], dtype=float)


    def elitist_unmatched(self, n: int, X: np.ndarray, y: np.ndarray) -> list(tuple(Classifier, np.ndarray)):
        """
        This method creates 'n' or less random classifiers with
        the inputs that were not matched by the 'solution_opt'
        elitist individual's active classifiers.
        If there are not enough unmatched points, then return
        only these points.
        """
        unmatched_points = np.zeros(X.shape[0])
        for cl in self.sol_opt.get_elitist().get_classifiers():
            unmatches = np.invert(cl.matches(X))
            unmatched_indices = np.logical_or(unmatched_points, unmatches)
        unmatched_points = X[unmatched_indices]

        classifiers = list()
        for point in unmatched_points:
            cl = Classifier.random_cl(X.shape[1], point=point)
            cl.fit(X, y)
            classifiers.append(cl)

        number_of_samples = min([n, len(classifiers)])
        return [ [cl, self.create_sigmas(X.shape[1])] for cl in Random().random.choice(classifiers, number_of_samples, False) ]


    def draw_examples_from_data(self, n: int, X: np.ndarray, y: np.ndarray) -> list(tuple(Classifier, np.ndarray)):
        """
        This method takes 'n' random examples out of the inputs and
        creates one classifier for each example taken.
        """
        start_tuples = list()
        idxs = Random().random.choice(np.arange(len(X)), n, False)
        for x in X[idxs]:
            cl = Classifier.random_cl(X.shape[1], point=x)
            cl.fit(X, y)
            start_tuples.append( [cl, self.create_sigmas(X.shape[1])] )
        return start_tuples


    def nondominated_sort(self, classifiers: list[Classifier], indexes: bool=True):
        """
        Takes a list of classifiers and returns all classifiers that were not
            dominated by any other in regard to error AND volume. This is
            equivalent to searching the pareto front

        Inspired by A Fast Elitist Non-Dominated Sorting GeneticAlgorithm for
            Multi-Objective Optimization: NSGA-II
            http://repository.ias.ac.in/83498/1/2-a.pdf

        If indexes is true, then the indexes of the nondominated classifiers
        in the list are return (instead of a new list)

        :param classifiers:
        :param indexes:
        :return:
        """
        candidates = list()
        candidates.append(classifiers[0] if not indexes else 0)
        for i in range(1, len(classifiers[1:])):
            cl = classifiers[i]
            volume_share_cl = cl.get_volume_share()
            to_be_added = False
            for j in range(len(candidates)):
                can = candidates[j] if not indexes else classifiers[candidates[j]]
                volume_share_can = can.get_volume_share()

                if can.error < cl.error and volume_share_can > volume_share_cl:
                    # classifier is dominated by this candidate and should not
                    # become a new candidate
                    to_be_added = False
                    break

                elif can.error > cl.error and volume_share_can < volume_share_cl:
                    # classifier dominates candidate
                    candidates.remove(can)
                    to_be_added = True

                else:
                    to_be_added = True

            if to_be_added:
                candidates.append(cl if not indexes else i)

        return candidates


class ES_OnePlusLambd(RuleDiscoverer):
    def __init__(self, pool: list[Classifier], solution_optimizer: SolutionOptimizer=None):
        super().__init__(pool, solution_optimizer)


    def step(self, X: np.ndarray, y: np.ndarray):
        nrules = Config().rule_discovery['nrules']
        start_cls, _ = self.create_start_tuples(nrules, X, y)

        for cl in start_cls:
            for i in range(Config().rule_discovery['steps_per_step']):
                children = list()
                for j in range(Config().rule_discovery['lmbd']):
                    child = deepcopy(cl)
                    child.mutate(Config().rule_discovery['sigma'])
                    child.fit(X, y)
                    children.append(child)
                ## ToDo instead of greedily taking the minimum, treating all
                ##  below a certain threshhold as equal might yield better  models
                #cl = children[np.argmin([child.get_weighted_error() for child in children])]
                cl = Random().random.choice(self.nondominated_sort(children))

            if cl.get_weighted_error() < Utilities.default_error(y[np.nonzero(cl.matches(X))]):
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


    def __init__(self, pool: list[Classifier], solution_optimizer: SolutionOptimizer=None) -> None:
        super().__init__(pool, solution_optimizer)
        self.sigmas = list()


    def step(self, X: np.ndarray, y: np.ndarray):
        generation_tuples = list()
        mu = Config().rule_discovery['mu']

        # create start points for evolutionary search (with mutation vectors)
        generation_tuples = self.create_start_tuples(mu, X, y)

        # steps forward in the evolutionary search
        for i in range(Config().rule_discovery['steps_per_step']):
            recmb_tuples      = self.recombine(generation_tuples)
            children_tuples   = self.mutate_and_fit(recmb_tuples, X, y)
            generation_tuples = self.replace(generation_tuples, children_tuples)

        # add search results to pool
        # mask = np.array([cl_tuple[0].get_weighted_error() < Utilities.default_error(y[np.nonzero(cl_tuple[0].matches(X))]) for cl_tuple in generation_tuples], dtype='bool')
        # filtered_tuples = np.array(generation_tuples, dtype=object)[mask]
        tuples_array = np.array(generation_tuples, dtype='object')
        nondominated_indexes = self.nondominated_sort(tuples_array[0,:], indexes=True)
        best_nondominated_tuples = tuples_array[nondominated_indexes]

        nondominated_classifiers_available = min(mu, len(nondominated_indexes))
        filtered_tuples = np.array(self.select_best_classifiers(best_nondominated_tuples, nondominated_classifiers_available), dtype=object)
        self.pool.extend( list(filtered_tuples[:,0]) )
        self.sigmas.extend( list(filtered_tuples[:,1]) )


    def recombine(self, parents_tuples: list[tuple[Classifier, np.ndarray]]) -> list[tuple[Classifier, np.ndarray]]:
        """
        This method decides which kind of recombination will be done,
        according to the rule discovery hyper parameter: 'recombination'
        If 'recombination' == 'i', then new 'lmbd' classifiers
        will be created out of the mean of 'rho' parents attributes.
        If 'recombination' == 'd', then the new classifiers are
        created randomly from the attributes from the parents
        (attributes do not cross values).
        If 'recombination' is None, then return the parents, and
        If it is somethin else, then only one classifier
        will be created and it will be a copy from one of the parents.
        """
        if len(parents_tuples) == 0:
            return None

        lmbd = Config().rule_discovery['lmbd']
        rho = Config().rule_discovery['rho']
        recombination_type = Config().rule_discovery['recombination']

        if recombination_type == 'i':
            return self.intermediate_recombination(parents_tuples, lmbd, rho)
        elif recombination_type == 'd':
            return self.discrete_recombination(parents_tuples, lmbd, rho)
        elif recombination_type is None:
            return parents_tuples
        else:
            cl_index = Random().random.choice(range(len(parents_tuples)))
            copied_tuple = deepcopy(parents_tuples[cl_index])
            return [copied_tuple]


    def intermediate_recombination(self, parents_tuples: list[tuple[Classifier, np.ndarray]], lmbd: int, rho: int) -> list[tuple[Classifier, np.ndarray]]:
        """
        This methods creates 'lmbd' classifiers from the average
        of 'rho' classifiers in parents.
        So, the new classifier look like this:
        Classifier( lowers=average(rho_candidates.lowerBounds),
                    uppers=average(rho_candidates.upperBounds), degree=1 )
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
            classifier = Classifier(lowers=avg_attrs[0], uppers=avg_attrs[1], degree=1)

            children_tuples.append((classifier, avg_attrs[2]))

        return children_tuples


    def discrete_recombination(self, parents_tuples: list[tuple[Classifier, np.ndarray]], lmbd: int, rho: int) -> list[tuple[Classifier, np.ndarray]]:
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
            classifier = Classifier(lowers=bounds[0], uppers=bounds[1], degree=1)
            children_tuples.append( (classifier, selected_attr[2]) )

        return children_tuples


    def mutate_and_fit(self, cls_tuples: list[tuple[Classifier, np.ndarray]], X: np.ndarray, y:np.ndarray) -> list[tuple[Classifier, np.ndarray]]:
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


    def replace(self, parents_tuples: list[tuple[Classifier, np.ndarray]], children_tuples: list[tuple[Classifier, np.ndarray]]) -> list[tuple[Classifier, np.ndarray]]:
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
    Page 13, Algorithm 4 - The (μ/μ, λ)-ES with Search Path
    Links:
    - PDF download: https://hal.inria.fr/hal-01155533/file/es-overview-2015.pdf
    - Refence: https://scholar.google.com/citations?user=NsIbm80AAAAJ&hl=en#
    """


    def __init__(self, pool: list[tuple[Classifier, np.ndarray]], solution_optimizer: SolutionOptimizer=None) -> None:
        super().__init__(pool, solution_optimizer)
        self.sigmas = list()


    def step(self, X: np.ndarray, y: np.ndarray):
        lmbd            = Config().rule_discovery['lmbd']
        mu              = Config().rule_discovery['mu']
        x_dim           = X.shape[1]
        sigma_coef      = np.sqrt(mu / (x_dim + mu))
        dist_global     = 1 + np.sqrt(mu / x_dim)
        dist_local      = 3 * x_dim
        tuples_for_pool = list()
        search_path     = 0
        start_point, _  = self.create_start_tuples(1, X, y)[0]
        start_sigma     = Random().random.normal(size=x_dim)

        for i in range(Config().rule_discovery['steps_per_step']):
            rnd_tuple_list = list()
            start_point.fit(X, y)
            print(f"reference weighted error: {start_point.get_weighted_error()}\tIn. step: {i}")

            # generating children with sigmas
            for j in range(lmbd):
                cl_sigmas = Random().random.normal(size=x_dim)
                cl = deepcopy(start_point)
                cl.lowerBounds, cl.upperBounds = np.stack((start_point.lowerBounds, start_point.upperBounds) + (start_sigma * cl_sigmas))
                cl.fit(X, y)
                rnd_tuple_list.append( [cl, cl_sigmas] )
            children_tuple_list = np.array(self.select_best_classifiers(rnd_tuple_list, mu), dtype=object)
            tuples_for_pool.extend( children_tuple_list )

            # recombination and parent update
            search_path = (1 - sigma_coef) * search_path + np.sqrt(sigma_coef * (2 - sigma_coef)) * (np.sqrt(mu) / mu) * np.sum(children_tuple_list[:,1])

            # expected value of a half normal distribution
            local_expected_value = np.sqrt(2 / np.pi)
            local_factor = np.power(( np.exp((np.abs(search_path) / local_expected_value) - 1) ),  (1 / dist_local))

            # There is an elegant way to replace Line 8b proposed by this articles at page 15.
            global_factor = np.power(( np.exp(( np.linalg.norm(search_path)**2 / x_dim ) - 1) ), (sigma_coef / dist_global) / 2)

            # step-size changes
            start_sigma = start_sigma * local_factor * global_factor

            # recombining parents attributes
            parents_attr = self.extract_classifier_attributes(children_tuple_list, x_dim)
            start_point.lowerBounds = np.mean(parents_attr[0], axis=0)
            start_point.upperBounds = np.mean(parents_attr[1], axis=0)

        # add children to pool
        tuples_array = np.array(tuples_for_pool, dtype='object')
        nondominated_indexes = self.nondominated_sort(tuples_array[0,:], indexes=True)
        best_nondominated_tuples = tuples_array[nondominated_indexes]

        nondominated_classifiers_available = min(mu, len(nondominated_indexes))
        filtered_tuples = np.array(self.select_best_classifiers(best_nondominated_tuples, nondominated_classifiers_available), dtype=object)
        self.pool.extend( list(filtered_tuples[:,0]) )
        self.sigmas.extend( list(filtered_tuples[:,1]) )


class ES_CMA(RuleDiscoverer):
    """
    This optimizer uses the Evolution Strategy
    Correlation Matrix Adaptation in order
    to promote diversity in the population.

    Its pool is a list of tuple, where each tuple
    has an classifier and a mutation vector (sigmas)
    associated with that classifier.

    Relevant hyper parameters are:
        'lmbd':             Number of new classifier generated.
                            After the 'lmbd' classifiers are
                            generated, only 'mu' will be selected
                            (according to the fitness/error).
                            Recommended value: 'lmbd' >= 5

        'mu':               Number of the best 'mu' new classifiers
                            that are going to be selected as parents
                            for the new classifier.
                            Recommended value: 'mu' = 'lmbd'/2

        'steps_per_step':   'steps_per_step'->  How many times we are going
                            to repeat the evolutionary search , when step()
                            is called. For instance, if steps_per_step
                            is 2, then run 2 steps in the evolutionary
                            search started by step().

    Implementation based on paper ES Overview 2015 by Hansen, Arnold & Auger.
    Page 15, Algorithm 5 - The (μ/μ_w, λ)-CMA-ES
    Links:
    - PDF download: https://hal.inria.fr/hal-01155533/file/es-overview-2015.pdf
    - Refence: https://scholar.google.com/citations?user=NsIbm80AAAAJ&hl=en#
    """


    def __init__(self, pool: list[Classifier], solution_optimizer: SolutionOptimizer=None) -> None:
        super().__init__(pool, solution_optimizer)
        self.sigmas = list()


    def step(self, X: np.ndarray, y: np.ndarray):
        lmbd            = Config().rule_discovery['lmbd']
        mu              = Config().rule_discovery['mu']
        x_dim           = X.shape[1]
        sigmas          = Random().random.normal(size=x_dim)
        sp_isotropic    = np.zeros(x_dim, dtype=float)
        sp_cov          = np.zeros(x_dim, dtype=float)
        C               = np.identity(x_dim)
        tuples_for_pool = list()
        start_point, _ = self.create_start_tuples(1, X, y)[0]

        for i in range(Config().rule_discovery['steps_per_step']):
            start_point.fit(X, y)
            print(f"reference weighted error: {start_point.get_weighted_error()}\tIn. step: {i}")

            # generating children with sigmas
            rnd_tuple_list = list()
            C_sqrt_diag = np.diag(np.sqrt(C))
            for j in range(lmbd):
                cl = deepcopy(start_point)
                # cl_sigmas = Random().random.uniform(low=0.25, high=1.25, size=x_dim)
                cl_sigmas = Random().random.normal(size=x_dim)
                cl.lowerBounds, cl.upperBounds = np.stack((start_point.lowerBounds, start_point.upperBounds) + ( (sigmas * np.sqrt(C)) @ cl_sigmas ))
                cl.fit(X, y)
                rnd_tuple_list.append( [cl, cl_sigmas] )
            children_tuple_list = np.array(self.select_best_classifiers(rnd_tuple_list, mu), dtype=object)
            tuples_for_pool.extend( children_tuple_list )

            # Initializing factors according to the children's weights
            children_weights = self.calculate_weights(children_tuple_list, lmbd)
            mu_weights = 1 / np.sum(children_weights**2)
            cov_isotropic = mu_weights / (x_dim + mu_weights)
            dist = 1 + np.sqrt(mu_weights / x_dim)
            cov_coef = (4 + mu_weights / x_dim) / (x_dim + 4 + 2 * mu_weights / x_dim)
            cov_one = 2 / (x_dim**2 + mu_weights)
            cov_mu = mu_weights / (x_dim**2 + mu_weights)
            cov_m = 1
            weighted_sigma_sum = np.sum(children_tuple_list[:,1] * children_weights)

            # start_point's boundaries update
            start_point.lowerBounds, start_point.upperBounds = np.stack((start_point.lowerBounds, start_point.upperBounds) + cov_m * sigmas * C_sqrt_diag * weighted_sigma_sum)
            start_point.fit(X, y)
            # search path isotropic update
            sp_isotropic = (1 - cov_isotropic) * sp_isotropic + np.sqrt(cov_isotropic * (2 - cov_isotropic)) * np.sqrt(mu_weights) * weighted_sigma_sum
            # search path with covariances update
            h_isotropic = 1 if (np.linalg.norm(sp_isotropic)**2 / x_dim) < 2 + 4 / (x_dim + 1) else 0
            weighted_cov_sigmas_sum = np.sum([ children_weights[i] * C_sqrt_diag * children_tuple_list[i,1] for i in range(mu) ])
            sp_cov = (1 - cov_coef) * sp_cov + h_isotropic * ( cov_coef * np.sqrt(mu_weights) * weighted_cov_sigmas_sum )
            # sigmas update
            sigmas *= np.power( np.exp( (np.linalg.norm(sp_isotropic)**2 / x_dim) - 1 ), ((cov_isotropic / dist) / 2) )
            # covariance matrix update
            cov_h = cov_one * (1 - h_isotropic**2) * cov_coef * (2 - cov_coef)
            weighted_cov_trans_sum = np.sum([ children_weights[i] * np.dot(C_sqrt_diag * children_tuple_list[i,1], (C_sqrt_diag * children_tuple_list[i,1]).T) for i in range(mu) ])
            C = (1 - cov_one + cov_h - cov_isotropic) * C + cov_one * np.dot(sp_cov, sp_cov.T) + cov_mu * weighted_cov_trans_sum

        # add children to pool
        tuples_array = np.array(tuples_for_pool, dtype='object')
        nondominated_indexes = self.nondominated_sort(tuples_array[0,:], indexes=True)
        best_nondominated_tuples = tuples_array[nondominated_indexes]

        nondominated_classifiers_available = min(mu, len(nondominated_indexes))
        filtered_tuples = np.array(self.select_best_classifiers(best_nondominated_tuples, nondominated_classifiers_available), dtype=object)
        self.pool.extend( list(filtered_tuples[:,0]) )
        self.sigmas.extend( list(filtered_tuples[:,1]) )


    def calculate_weights(self, cls_tuples: list[tuple[Classifier, np.ndarray]], lmbd: int) -> np.ndarray:
        """
        This method is designed to calculate the w_k
        in the original algorithm. It is a function, that
        returns the weights of the children, where
        w_k = w(k) / Sum(w(k)) (from k to µ)
        But, in order to avoid repeating this calculation,
        the implemented method return an array from all
        classifiers.

        The selection enviroment is a truncation selection,
        but instead of the rank directly, the log from it
        is used.
        """
        tuples_array = np.array(cls_tuples, dtype=object)
        if tuples_array.ndim == 1:
            weighted_errors = np.array([ cl_tuple[0].get_weighted_error() for cl_tuple in cls_tuples ], dtype=float)
        else:
            weighted_errors = np.array([ cl.get_weighted_error() for cl in tuples_array[:,0] ], dtype=float)
        ranked_indexes = np.argsort(weighted_errors)
        weights = np.ones(tuples_array.shape[0], dtype=float)

        for i in range(tuples_array.shape[0]):
            # Klaus: Research log
            weights[i] = np.log(lmbd/2 + 0.5) - np.log(ranked_indexes[i] + 1)

        return weights
