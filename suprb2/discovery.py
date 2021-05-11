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
    def __init__(self, pool: List[Classifier]):
        self.pool = pool


    def step(self, X: np.ndarray, y: np.ndarray):
        pass


    def extract_classifier_attributes(self, classifiers: List[Classifier], x_dim: int, rho: int=None, row: int=None, sigmas: bool=False):
        """
        Creates an array with shape (2, rho, x_dim),
        where 2 is the number of relevant attributes (lowers and uppers).
        If 'sigmas' == True, then sigmas is also added to the relevant
        attributes (lowers, uppers, sigmas).
        """
        rho = rho if rho is not None else len(classifiers)
        rows = 3 if sigmas is True else 2
        classifier_attrs = np.zeros((rows, rho, x_dim), dtype=float)
        for i in range(rho):
            cl = classifiers[i]
            classifier_attrs[0][i] = cl.lowerBounds
            classifier_attrs[1][i] = cl.upperBounds
            if sigmas:
                classifier_attrs[2][i] = self.create_sigmas(x_dim, cl)

        return classifier_attrs if row is None else classifier_attrs[row]


    def create_sigmas(self, x_dim: int, cl: Classifier=None):
        """
        Creates an array with size 'x_dim' with values
        from a normal distribution.
        """
        return Random().random.normal(size=x_dim)


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


    def __init__(self, pool: List[Classifier]):
        super().__init__(pool)
        self.sigmas_dict = {}


    def step(self, X: np.ndarray, y: np.ndarray):
        generation = []
        mu = Config().rule_discovery['mu']

        # create start points for evolutionary search (with mutation vectors)
        idxs = Random().random.choice(np.arange(len(X)), mu, False)
        for x in X[idxs]:
            cl = Classifier.random_cl(x, X.shape[1])
            cl.fit(X, y)
            generation.append(cl)

        gen_sigmas = self.extract_classifier_attributes(generation, rho=mu, x_dim=X.shape[1], row=2)
        # steps forward in the evolutionary search
        for i in range(Config().rule_discovery['steps_per_step']):
            recmb_classifiers, recmb_sigmas = self.recombine(generation, gen_sigmas)
            children, children_sigmas = self.mutate_and_fit(recmb_classifiers, X, y, recmb_sigmas)
            generation, gen_sigmas = self.replace(generation, gen_sigmas, children, children_sigmas)

        # add search results to pool
        mask = np.array([cl.get_weighted_error() < Utilities.default_error(y[np.nonzero(cl.matches(X))]) for cl in generation], dtype='bool')
        new_classifiers = np.array(generation, dtype='object')[mask]
        new_sigmas = np.array(gen_sigmas, dtype=float)[mask]
        self.pool.extend(new_classifiers)
        # after adding classifiers to the pool, register them with the correspondent sigmas vector
        self.register_classifier_sigmas(new_classifiers, new_sigmas)


    def recombine(self, parents: List[Classifier], sigmas: np.ndarray):
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

        This function returns the recombined children and their sigmas:
        children, children_sigmas = ES_MuLambd().recombine(parents, parents_sigmas)
        """
        if len(parents) == 0:
            return ([], np.array([], dtype=float))

        lmbd = Config().rule_discovery['lmbd']
        rho = Config().rule_discovery['rho']
        recombination_type = Config().rule_discovery['recombination']

        if recombination_type == 'intermediate':
            return self.intermediate_recombination(parents, lmbd, rho, sigmas)
        elif recombination_type == 'discrete':
            return self.discrete_recombination(parents, lmbd, rho, sigmas)
        else:
            cl_index = Random().random.choice(range(len(parents)))
            copied_cl, copied_sigmas = deepcopy((parents[cl_index], sigmas[cl_index]))
            return [copied_cl], copied_sigmas


    def intermediate_recombination(self, parents: List[Classifier], lmbd: int, rho: int, sigmas: np.ndarray):
        """
        This methods creates 'lmbd' classifiers from the average
        of 'rho' classifiers in parents.
        So, the new classifier look like this:
        Classifier( lowers=average(rho_candidates.lowerBounds),
                    uppers=average(rho_candidates.upperBounds),
                    sigmas=average(rho_candidates.sigmas),
                    LinearRegression(), degree=1 )

        The return values are a tuple of the new generated classifiers
        and their respective sigmas:
        children, children_sigmas = ES_MuLambd().intermediate_recombination(parents, parents_sigmas)
        """
        children_cls, children_sigmas = [], []
        x_dim = parents[0].lowerBounds.size if type(parents[0].lowerBounds) == np.ndarray else 1
        for i in range(lmbd):
            candidates = Random().random.choice(parents, rho, False)
            classifier_attrs = self.extract_classifier_attributes(candidates, x_dim=x_dim, rho=rho)

            avg_attrs = np.array([[np.mean(classifier_attrs[0].flatten())],
                                  [np.mean(classifier_attrs[1].flatten())],
                                  [np.mean(classifier_attrs[2].flatten())]])

            classifier = Classifier(avg_attrs[0], avg_attrs[1], LinearRegression(), 1) # Reminder: LinearRegression might change in the future
            children_sigmas = np.append(children_sigmas, avg_attrs[2])
            children_cls.append(classifier)

        return children_cls, children_sigmas


    def discrete_recombination(self, parents: List[Classifier], lmbd: int, rho: int, sigmas: np.ndarray):
        """
        This method creates 'lmbd' classifiers picking randomly
        the attributes from 'rho' parents. The values do not cross
        types (so, values used for upperBounds, can only be used for
        one classifier's upperBounds).
        If the values are flipped (lowerBounds > upperBounds), then
        unflip it and save.

        The return values are a tuple of the new generated classifiers
        and their respective sigmas:
        children, children_sigmas = ES_MuLambd().discrete_recombination(parents, parents_sigmas)
        """
        children_cls, children_sigmas = [], []
        x_dim = parents[0].lowerBounds.size if type(parents[0].lowerBounds) == np.ndarray else 1
        for i in range(lmbd):
            candidates = Random().random.choice(parents, rho, False)
            classifier_attrs = self.extract_classifier_attributes(candidates, x_dim=x_dim, rho=rho)
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
            children_sigmas = np.append(children_sigmas, selected_attr[2])
            children_cls.append(classifier)

        return children_cls, children_sigmas


    def mutate_and_fit(self, classifiers: List[Classifier], X: np.ndarray, y:np.ndarray, classifiers_sigmas: np.ndarray):
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

        The return values are a tuple of the new generated classifiers
        and their respective sigmas:
        children, children_sigmas = ES_MuLambd().mutate_and_fit(parents, parents_sigmas)
        """
        children_cls, children_sigmas = [], []
        global_learning_rate = Random().random.normal(scale=Config().rule_discovery['global_tau'])
        cls_len = len(classifiers)

        for i in range(cls_len):
            cl = classifiers[i]

            # Apply global and local learning factors to the classifier's mutation vector
            local_learning_rate = Config().rule_discovery['local_tau'] * Random().random.normal(size=X.shape[1])
            sigmas = classifiers_sigmas[i]
            mutated_sigmas = sigmas * np.exp(local_learning_rate + global_learning_rate)

            # Mutate classifier's matching function
            lowers = Random().random.normal(loc=cl.lowerBounds, scale=mutated_sigmas, size=X.shape[1])
            uppers = Random().random.normal(loc=cl.upperBounds, scale=mutated_sigmas, size=X.shape[1])
            lu = np.clip(np.sort(np.stack((lowers, uppers)), axis=0), a_max=1, a_min=-1)
            cl.lowerBounds = lu[0]
            cl.upperBounds = lu[1]
            # cl.mutate(Config().rule_discovery['sigma'])
            cl.fit(X, y)
            if cl.get_weighted_error() < Utilities.default_error(y[np.nonzero(cl.matches(X))]):
                children_cls.append(cl)
                children_sigmas = np.append(children_sigmas, mutated_sigmas)
        return children_cls, np.array(children_sigmas, ndmin=1)


    def replace(self, parents: List[Classifier], parents_sigmas: np.ndarray, children: List[Classifier], children_sigmas: np.ndarray):
        if Config().rule_discovery['replacement'] == '+':
            return (parents + children), np.append(parents_sigmas, children_sigmas)
        else:
            return children, children_sigmas


    def create_sigmas(self, x_dim: int, cl: Classifier=None):
        """
        Returns the mutation vector associated with the
        given classifier ('cl').
        If 'cl' is not in this optimzer's pool, then return
        x_dim random values from a Guassian distribution:
        cl.sigmas[x_dim] ~ N(0, 1).

        IMPORTANT: The values for new classifiers are not
        persistent. If you need it, then ensure that it is
        saved in an array.
        """
        try:
            cl_idx = self.pool.index(cl)
            return self.sigmas_dict[cl_idx]
        except ValueError:
            return np.abs(Random().random.normal(size=x_dim))


    def register_classifier_sigmas(self, classifiers: List[Classifier], sigmas: np.ndarray):
        """
        Saves the given classifier's sigmas vector in this optimzer
        instance dictionary (if there is one).

        It only registers the classifier, if the same is in the
        optimizer's pool. After registration True is returned.

        If the classifier is not in the pool, then return False.
        """
        try:
            cls_len = len(classifiers)
            for i in range(cls_len):
                index = self.pool.index(classifiers[i])
                self.sigmas_dict.update([(index, sigmas[i])])
            return True
        except ValueError:
            return False


    def extract_classifier_attributes(self, classifiers: List[Classifier], x_dim: int, rho: int=None, row: int=None):
        return super().extract_classifier_attributes(classifiers, x_dim, rho, row, sigmas=True)


class ES_CSA(RuleDiscoverer):
    """
    This optimizer uses the Evolutio Strategy
    Cumullative Step-Size Adaptation in order
    to promote diversity in the population.
    Relevant hyper parameters are:
        'lmbd':         Number of new classifier generated.
                        After the 'lmbd' classifiers are
                        generated, only 'mu' will be selected
                        (according to the fitness/error).
                        Recommended value: positive int

        'mu':           Number of the best 'mu' new classifiers
                        that are going to be selected as parents
                        for the new classifier.
                        Recommended value: 'lmbd'/4

    'steps_per_step':   'steps_per_step'->  How many times we are going
                        to repeat the evolutionary search , when step()
                        is called. For instance, if steps_per_step
                        is 2, then run 2 steps in the evolutionary
                        search started by step().
    """


    def __init__(self, pool: List[Classifier]):
        super().__init__(pool)


    def step(self, X: np.ndarray, y: np.ndarray):
        lmbd            = Config().rule_discovery['lmbd']
        mu              = Config().rule_discovery['mu']
        x_dim           = X.shape[1]
        sigma_coef      = np.sqrt(mu / (x_dim + mu))
        dist_global     = 1 + np.sqrt(mu / x_dim)
        dist_local      = 3 * x_dim
        new_cl_tuple    = [Classifier.random_cl(x_dim), self.create_sigmas(x_dim)]
        search_path     = 0

        for i in range(Config().rule_discovery['steps_per_step']):
            rnd_tuple_list = list()

            # generating parents with sigmas
            for j in range(lmbd):
                cl = deepcopy(new_cl_tuple[0])
                cl.fit(X, y)
                rnd_tuple_list.append([cl, self.create_sigmas(x_dim)])
            parent_tuple_list = np.array(self.select_best_classifiers(rnd_tuple_list, mu))

            # updating search_path and step-sizes vector
            search_path = (1 - sigma_coef) * search_path + np.sqrt(sigma_coef * (2 - sigma_coef)) * (np.sqrt(mu) / mu) * np.sum(parent_tuple_list[:,1])
            # local changes
            E = np.abs(Random().random.normal())
            local_factor = np.power(( np.exp((np.abs(search_path) / E) - 1) ),  (1 / dist_local))
            # global changes
            E_vector = np.linalg.norm(self.create_sigmas(x_dim))
            global_factor = np.power(( np.exp((np.absolute(search_path) / E_vector) - 1) ), (sigma_coef / dist_global))
            # step-size changes
            new_cl_tuple[1] = new_cl_tuple[1] * local_factor * global_factor

            # recombining parents attributes
            parents_attr = self.extract_classifier_attributes(parent_tuple_list[:,0], x_dim)
            new_cl_tuple[0].lowerBounds = (1/mu) * np.sum(parents_attr[0])
            new_cl_tuple[0].upperBounds = (1/mu) * np.sum(parents_attr[1])

        # add tuple to pool
        self.pool.append( new_cl_tuple )


    def select_best_classifiers(self, tuple_list: List[Tuple[Classifier, np.ndarray]], mu: int):
        classifiers, cls_sigmas = list(zip(*tuple_list))
        idx = np.argpartition([ cl.get_weighted_error() for cl in classifiers ], mu).astype(int)[:mu]
        return list(zip(*[np.array(classifiers)[idx], np.array(cls_sigmas)[idx]]))

    def extract_classifier_attributes(self, classifiers: List[Classifier], x_dim: int, rho: int=None, row: int=None):
        return super().extract_classifier_attributes(classifiers, x_dim, rho, row, sigmas=True)
