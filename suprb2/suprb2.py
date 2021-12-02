import warnings
from collections import defaultdict
from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import clone
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array

from .base import BaseRegressor
from .exceptions import PopulationEmptyWarning
from .individual import Individual
from .logging import BaseLogger
from .optimizer.individual import IndividualOptimizer
from .optimizer.individual.ga import GeneticAlgorithm
from .optimizer.rule import RuleGeneration
from .optimizer.rule.es import ES1xLambda
from .rule import Rule
from .utils import check_random_state, spawn_random_states, estimate_bounds, flatten


class SupRB2(BaseRegressor):
    """ The multi-solution batch learning LCS developed by the Organic Computing group at Universität Augsburg.

    Parameters
    ----------
    rule_generation: RuleGeneration
        Optimizer used to evolve the :class:`Rule`s. If None is passed, it is set to :class:`ES1xLambda`.
    individual_optimizer: IndividualOptimizer
        Optimizer used to evolve the :class:`Individual`s. If None is passed, it is set to :class:`GeneticAlgorithm`.
    n_iter: int
        Iterations the LCS will perform.
    n_initial_rules: int
        Number of :class:`Rule`s generated before the first step.
    n_rules: int
        Number of :class:`Rule`s generated in the every step.
    random_state : int, RandomState/Generator instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
    verbose : int
        - >0: Show some description.
        - >5: Show elaborate description.
        - >10: show all
    logger: BaseLogger
    n_jobs: int
        The number of threads / processes the fitting process uses.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
        None is a marker for 'unset' that will be interpreted as n_jobs=1 (sequential execution) unless the call is
        performed under a parallel_backend context manager that sets another value for n_jobs.
        Taken from the `joblib.Parallel` documentation.
    """

    step_: int = 0
    total_rules_: int

    pool_: list[Rule]
    elitist_: Individual
    random_state_: np.random.Generator
    random_state_seeder_: np.random.SeedSequence

    rule_generation_: RuleGeneration
    individual_optimizer_: IndividualOptimizer

    fit_metrics_: Union[defaultdict, pd.DataFrame]

    n_features_in_: int

    logger_: BaseLogger

    def __init__(self,
                 rule_generation: RuleGeneration = None,
                 individual_optimizer: IndividualOptimizer = None,
                 n_iter: int = 16,
                 n_initial_rules: int = 0,
                 n_rules: int = 16,
                 random_state: int = None,
                 verbose: int = 0,
                 logger: BaseLogger = None,
                 n_jobs: int = 1,
                 ):
        self.n_iter = n_iter
        self.n_initial_rules = n_initial_rules
        self.n_rules = n_rules
        self.rule_generation = rule_generation
        self.individual_optimizer = individual_optimizer
        self.random_state = random_state
        self.verbose = verbose
        self.logger = logger
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray, cleanup=False):
        """ Fit SupRB.2.

            Parameters
            ----------
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples,) or (n_samples, n_outputs)
                The target values.
            cleanup : bool
                Optional cleanup of unused rules after fitting.

            Returns
            -------
            self : BaseEstimator
                Returns self.
        """

        # Check that x and y have correct shape
        X, y = check_X_y(X, y, dtype='float64', y_numeric=True)
        y = check_array(y, ensure_2d=False, dtype='float64')

        # Init sklearn interface
        self.n_features_in_ = X.shape[1]

        # Initialise components
        self.pool_ = []

        self._validate_rule_generation(default=ES1xLambda())
        self._validate_individual_optimizer(default=GeneticAlgorithm())

        self._propagate_component_parameters()
        self._init_bounds(X)

        # Init optimizers
        self.individual_optimizer_.pool_ = self.pool_

        # Random state
        self.random_state_ = check_random_state(self.random_state)
        self.random_state_seeder_ = np.random.SeedSequence(self.random_state)

        # Init Logging
        self.logger_ = clone(self.logger) if self.logger is not None else None
        if self.logger_ is not None:
            self.logger_.log_init(X, y, self)

        with Parallel(n_jobs=self.n_jobs) as parallel:
            # Fill population before first step
            if self.n_initial_rules > 0:
                self._generate_rules(X, y, self.n_initial_rules, parallel=parallel)

            # Main loop
            for self.step_ in range(self.n_iter):
                # Insert new rules into population
                self._generate_rules(X, y, self.n_rules, parallel=parallel)

                # Optimize individuals
                self._select_rules(X, y)

                # Log Iteration
                if self.logger_ is not None:
                    self.logger_.log_iteration(X, y, self, iteration=self.step_)

        if cleanup:
            self._cleanup()

        self.elitist_ = self.individual_optimizer_.elitist()
        self.is_fitted_ = True

        # Log final result
        if self.logger_ is not None:
            self.logger_.log_final(X, y, self)

        return self

    @staticmethod
    def _generate_rule_with_mean(X: np.ndarray, y: np.ndarray, rule_generation: RuleGeneration, mean: np.ndarray,
                                 random_state: np.random.RandomState) -> Rule:
        """
        Helper method to execute several `RuleGeneration`s in parallel.
        Generates rules from some distribution with mean `mean`.
        """
        rule_generation.mean = mean
        rule_generation.random_state = random_state
        return rule_generation.optimize(X, y)

    def _generate_rules(self, X: np.ndarray, y: np.ndarray, n_rules: int, parallel: Parallel = None):
        """Performs the rule discovery / rule generation (RG) process."""

        self._log_to_stdout(f"Discovering {n_rules} rules", priority=4)

        # TODO: Really generate n_rules
        # For now, less rules could be generated because some are not accepted into the population

        # Bias the indices that input values that were matched less than others
        # have a higher probability to be selected
        if self.pool_:
            counts = np.count_nonzero(np.stack([rule.match_ for rule in self.pool_], axis=0) == 0, axis=0)
            counts_sum = np.sum(counts)
            # If all input values are matched by every rule, no bias is needed
            probabilities = counts / counts_sum if counts_sum != 0 else None
        else:
            # No bias needed when no rule exists
            probabilities = None

        indices = self.random_state_.choice(np.arange(len(X)), n_rules, p=probabilities)

        # Init every optimizer with own random state and generate the rules in parallel
        random_states = spawn_random_states(self.random_state_seeder_, len(indices))
        result = parallel(delayed(self._generate_rule_with_mean)(X, y, clone(self.rule_generation_), mean, random_state)
                          for mean, random_state in zip(X[indices], random_states))

        # Flatten the result (may be nested) and filter all invalid rules
        new_rules = filter(lambda rule: rule is not None, flatten(result))

        # Extend the pool with the new rules
        self.pool_.extend(new_rules)

        if not self.pool_:
            warnings.warn(
                "The population is empty, even after generating rules. "
                "Individual optimization will be skipped.",
                PopulationEmptyWarning)

    def _select_rules(self, X: np.ndarray, y: np.ndarray):
        """Performs rule selection (RS)."""

        self._log_to_stdout(f"Optimizing populations", priority=4)

        self.individual_optimizer_.optimize(X, y)

    def predict(self, X: np.ndarray):
        # Check is fit had been called
        check_is_fitted(self, ['is_fitted_'])
        # Input validation
        X = check_array(X)

        return self.elitist_.predict(X)

    def _validate_rule_generation(self, default=None):
        self.rule_generation_ = clone(self.rule_generation) if self.rule_generation is not None else clone(default)

    def _validate_individual_optimizer(self, default=None):
        self.individual_optimizer_ = clone(self.individual_optimizer) \
            if self.individual_optimizer is not None else clone(default)

    def _log_to_stdout(self, message, priority=1):
        if self.verbose >= priority:
            message = f"[{self.step_ + 1}/{self.n_iter}] {message}"
            print(message)

    def _propagate_component_parameters(self):
        """Propagate shared parameters to subcomponents."""
        keys = ['random_state', 'n_jobs']
        params = {key: value for key, value in self.get_params().items() if key in keys}

        self.rule_generation_.set_params(**params)
        self.individual_optimizer_.set_params(**params)

    def _init_bounds(self, X):
        """Try to estimate all bounds that are not provided at the start from the training data."""
        bounds = estimate_bounds(X)
        for key, value in self.rule_generation_.get_params().items():
            if key.endswith('bounds') and value is None:
                self._log_to_stdout(f"Found empty bounds for {key}, estimating from data")
                self.rule_generation_.set_params(**{key: bounds})

    def _cleanup(self):
        """Optional cleanup of unused rules after fitting."""
        self.pool_ = self.individual_optimizer_.elitist().subpopulation

        self.individual_optimizer_.elitist().genome = np.ones(len(self.pool_), dtype='bool')
        self.individual_optimizer_.elitist().pool = self.pool_

    def _more_tags(self):
        """
        For scikit-learn compatability, as it shows R^2 < 0.5 on certain data sets.
        Needed so that the model passes `check_estimator()`.
        """
        return {
            'poor_score': True,
        }
