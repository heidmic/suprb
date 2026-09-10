import warnings
import traceback

import numpy as np
from sklearn import clone
from sklearn.utils import check_X_y
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted, check_array, validate_data


from .base import BaseRegressor
from .exceptions import PopulationEmptyWarning
from .solution import Solution
from .logging import BaseLogger, DefaultLogger
from .optimizer.solution import SolutionComposition, ga
from .optimizer.solution.ga import GeneticAlgorithm
from .optimizer.rule import RuleDiscovery
from .optimizer.rule.es import ES1xLambda
from .rule import Rule
from .rule.matching import MatchingFunction, OrderedBound, GABIL, MixedTypes
from .utils import check_random_state, estimate_bounds
from .solution.mixing_model import ErrorExperienceHeuristic
from .solution.fitness import PseudoBIC


class SupRB(BaseRegressor):
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.single_output = False
        tags.non_deterministic = True
        return tags

    def _more_tags(self):
        # additional or override tags
        return {
            # 'some_tag': True,
        }

    """The multi-solution batch learning LCS developed by the Organic Computing group at Universität Augsburg.

    Parameters
    ----------
    rule_discovery: RuleDiscovery
        Optimizer used to evolve the :class:`Rule`s. If None is passed, it is set to :class:`ES1xLambda`.
    solution_composition: SolutionComposition
        Optimizer used to evolve the :class:`Solution`s. If None is passed, it is set to :class:`GeneticAlgorithm`.
    n_iter: int
        Iterations the LCS will perform.
    n_initial_rules: int
        Number of :class:`Rule`s generated before the first step. Note that 
        n_initial_rules + n_rules will be created before the first elitist is 
        selected using solution composition.
    n_rules: int
        Number of :class:`Rule`s generated in the every step.
    random_state : int, RandomState/Generator instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
    verbose : int
        - >0: Show some description.
        - >5: Show elaborate description.
        - >10: show all
    logger: BaseLogger
        If none is set it defaults to DefaultLogger()
    n_jobs: int
        The number of threads / processes the fitting process uses.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
        None is a marker for 'unset' that will be interpreted as n_jobs=1 (sequential execution) unless the call is
        performed under a parallel_backend context manager that sets another value for n_jobs.
        Taken from the `joblib.Parallel` documentation.
    early_stopping_patience: int
        Sets the patience for how many iteration we try to find a better result before we do an early stopping (-1 disabling the early stopping).
    early_stopping_delta: int
        The current fitness needs to be higher than this delta of the previous iteration fitness to be considered a "better" iteration
    """

    step_: int = 0

    pool_: list[Rule]
    elitist_: Solution

    random_state_: np.random.Generator

    rule_discovery_: RuleDiscovery
    rule_discovery_seeds_: list[np.random.SeedSequence]
    initial_rule_seeds_: list[np.random.SeedSequence]

    solution_composition_: SolutionComposition
    solution_composition_seeds_: list[np.random.SeedSequence]

    matching_type_: MatchingFunction

    n_features_in_: int

    logger_: BaseLogger

    def __init__(
        self,
        rule_discovery: RuleDiscovery = None,
        solution_composition: SolutionComposition = None,
        matching_type: MatchingFunction = None,
        categorical_encoder=None,
        encode_categoricals: bool = False,
        n_iter: int = 32,
        n_initial_rules: int = 0,
        n_rules: int = 4,
        random_state: int = None,
        verbose: int = 1,
        logger: BaseLogger = None,
        n_jobs: int = 1,
        early_stopping_patience: int = -1,
        early_stopping_delta: float = 0,
    ):
        self.n_iter = n_iter
        self.n_initial_rules = n_initial_rules
        self.n_rules = n_rules
        self.matching_type = matching_type
        self.categorical_encoder = categorical_encoder
        self.encode_categoricals = encode_categoricals
        self.rule_discovery = rule_discovery
        self.solution_composition = solution_composition
        self.random_state = random_state
        self.verbose = verbose
        self.logger = logger
        self.n_jobs = n_jobs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta

    def check_early_stopping(self):
        if self.early_stopping_patience > 0:
            fitness_diff = self.solution_composition_.elitist().fitness_ - self.previous_fitness_

            if fitness_diff > self.early_stopping_delta:
                self.early_stopping_counter_ = 0
            else:
                self.early_stopping_counter_ += 1
                if self.early_stopping_patience <= self.early_stopping_counter_:
                    print(
                        f"Execution was stopped early after {self.early_stopping_patience} cycles with no significant changes."
                    )
                    print(f"The elitist fitness value was: {self.previous_fitness_}")
                    return True
        return False

    def fit(self, X: np.ndarray, y: np.ndarray, cleanup=False):
        """Fit SupRB.2.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values.
        cleanup : bool
            Optional cleanup of unused rules and components after fitting. Can be used to reduce size if only the
            final model is relevant. Note that all information about the fitting process itself is removed.
        Returns
        -------
        self : BaseEstimator
            Returns self.
        """

        # Set these values so we gracefully exit on error
        self.early_stopping_counter_ = 0
        self.previous_fitness_ = 0
        self.is_error_ = False
        self.elitist_ = Solution([0, 0, 0], [0, 0, 0], ErrorExperienceHeuristic(), PseudoBIC())
        self.elitist_.fitness_ = 0
        self.elitist_.error_ = 99999
        self.elitist_.complexity_ = 99999

        # Check that x and y have correct shape
        X = self._encode_categoricals(X, fitting=True)
        X, y = validate_data(self, X, y, ensure_2d=True)

        # Init sklearn interface
        self.n_features_in_ = X.shape[1]

        # Random state
        self.random_state_ = check_random_state(self.random_state)
        seeds = np.random.SeedSequence(self.random_state).spawn(self.n_iter * 2 + 1)
        self.rule_discovery_seeds_ = seeds[:-1:2]
        self.solution_composition_seeds_ = seeds[1::2]
        self.initial_rule_seeds_ = seeds[-1:]

        # Initialise components
        self.pool_ = []

        self._validate_rule_discovery(default=ES1xLambda())
        self._validate_solution_composition(default=GeneticAlgorithm())
        self._validate_matching_type(default=OrderedBound(np.array([])))
        self._validate_logger(default=DefaultLogger())

        self._propagate_component_parameters()
        self._init_bounds(X)
        self._init_levels(X)
        self._init_matching_type()

        # Init optimizers
        self.solution_composition_.pool_ = self.pool_
        self.solution_composition_.init.fitness.max_genome_length_ = self.n_rules * self.n_iter + self.n_initial_rules
        self.rule_discovery_.pool_ = self.pool_

        # Init Logging
        self.logger_.log_init(X, y, self)

        # Fill population before first step
        if self.n_initial_rules > 0:
            if self._catch_errors(self._discover_rules, X, y, initial=True):
                return self

        # Main loop
        for self.step_ in range(self.n_iter):
            # Insert new rules into population
            if self._catch_errors(self._discover_rules, X, y, initial=False):
                return self

            # Optimize solutions
            if self._catch_errors(self._compose_solution, X, y):
                return self

            # Log Iteration
            self.logger_.log_iteration(X, y, self, iteration=self.step_)

            if self.check_early_stopping():
                break

            self.previous_fitness_ = self.solution_composition_.elitist().fitness_

        self.elitist_ = self.solution_composition_.elitist().clone()
        self.is_fitted_ = True

        # Log final result
        self.logger_.log_final(X, y, self)

        if cleanup:
            self._cleanup()

        return self

    def _catch_errors(self, func, X, y, **kwargs):
        try:
            func(X, y, **kwargs)
            return False
        except ValueError as e:
            # Capture the full traceback and print it
            tb = traceback.format_exc()
            warnings.warn(f"The following ValueError has occurred:\n{e}\nTraceback:\n{tb}")
            self.is_fitted_ = True
            self.is_error_ = True
            return True
        except Exception as e:
            # Capture the full traceback and print it
            tb = traceback.format_exc()
            warnings.warn(f"An error has occurred. This is likely due to a bad configuration:\n{e}\nTraceback:\n{tb}")
            self.is_fitted_ = True
            self.is_error_ = True
            return True

    def _discover_rules(self, X: np.ndarray, y: np.ndarray, initial: bool):
        """Performs the rule discovery / rule generation (RG) process."""

        n_rules = self.n_initial_rules if initial else self.n_rules

        self._log_to_stdout(f"Generating {n_rules} rules", priority=4)

        # Update the current elitist
        self.rule_discovery_.elitist_ = self.solution_composition_.elitist()

        # Update the random state
        if initial:
            self.rule_discovery_.random_state = self.initial_rule_seeds_[0]
        else:
            self.rule_discovery_.random_state = self.rule_discovery_seeds_[self.step_]

        # Generate new rules
        new_rules = self.rule_discovery_.optimize(X, y, n_rules=n_rules)

        # Extend the pool with the new rules
        self.pool_.extend(new_rules)

        if not self.pool_:
            warnings.warn(
                "The population is empty, even after generating rules. " "Solution optimization will be skipped.",
                PopulationEmptyWarning,
            )

    def _compose_solution(self, X: np.ndarray, y: np.ndarray):
        """Performs solution composition."""

        self._log_to_stdout(f"Optimizing populations", priority=4)

        # Update the random state
        self.solution_composition_.random_state = self.solution_composition_seeds_[self.step_]
        # Optimize
        self.solution_composition_.optimize(X, y)

    def predict(self, X):
        check_is_fitted(self)
        X = self._encode_categoricals(X, fitting=False)
        X = validate_data(self, X, ensure_2d=True, reset=False)

        if hasattr(self, "is_error_") and self.is_error_:
            return [0] * len(X)
        else:
            return self.elitist_.predict(X)

    def _validate_rule_discovery(self, default=None):
        self.rule_discovery_ = clone(self.rule_discovery) if self.rule_discovery is not None else clone(default)

    def _validate_solution_composition(self, default=None):
        self.solution_composition_ = (
            clone(self.solution_composition) if self.solution_composition is not None else clone(default)
        )

    def _validate_matching_type(self, default=None):
        self.matching_type_ = clone(self.matching_type) if self.matching_type is not None else clone(default)

    def _validate_logger(self, default=None):
        self.logger_ = clone(self.logger) if self.logger is not None else clone(default)

    def _log_to_stdout(self, message, priority=1):
        if self.verbose >= priority:
            message = f"[{self.step_ + 1}/{self.n_iter}] {message}"
            print(message)

    def _propagate_component_parameters(self):
        """Propagate shared parameters to subcomponents."""
        keys = ["n_jobs"]
        params = {key: value for key, value in self.get_params().items() if key in keys}

        self.rule_discovery_.set_params(**params)
        self.solution_composition_.set_params(**params)

    def _init_bounds(self, X):
        """Try to estimate all bounds that are not provided at the start from the training data."""
        bounds = estimate_bounds(X)
        for key, value in self.rule_discovery_.get_params().items():
            if key.endswith("bounds") and value is None:
                self._log_to_stdout(f"Found empty bounds for {key}, estimating from data")
                self.rule_discovery_.set_params(**{key: bounds})

    def _categorical_mask(self):
        """Which input columns are categorical, or None if none are.

        The mask lives on the matching function, since that is what decides
        how each attribute is interpreted. A plain GABIL treats every column
        as categorical; MixedTypes carries an explicit mask.
        """
        matching_type = self.matching_type

        if isinstance(matching_type, MixedTypes):
            return np.asarray(matching_type.mask, dtype=bool)

        if isinstance(matching_type, GABIL):
            return None

        return None

    def _encode_categoricals(self, X, fitting: bool):
        """Encode raw categorical columns to integer codes.

        SupRB works on numeric codes throughout, so raw labels have to be
        converted before sklearn's input validation, which would otherwise
        reject them. The encoder is fitted once during fit and only applied
        afterwards, so that a label keeps the same code at predict time. An
        encoder refitted on the test data would silently assign different
        codes to the same categories.
        """
        if not self.encode_categoricals:
            return X

        mask = self._categorical_mask()
        if mask is None or not mask.any():
            return X

        X = np.asarray(X, dtype=object)
        columns = X[:, mask]

        if fitting:
            self.fitted_encoder_ = OrdinalEncoder()
            encoded = self.fitted_encoder_.fit_transform(columns)
        else:
            check_is_fitted(self, "fitted_encoder_")
            encoded = self.fitted_encoder_.transform(columns)

        X = X.copy()
        X[:, mask] = encoded

        return X.astype(float)

    def _init_levels(self, X: np.ndarray):
        """Give the categorical half its level counts.

        A bitstring needs to know how many levels the attribute has before
        any rule can be built for it. That is a property of the dataset, so
        it is resolved here rather than guessed inside the matching
        function.

        Three sources are used, in order of how much they can be trusted:
        an encoder supplied by the user, which knows the full vocabulary
        including levels absent from this particular split; otherwise the
        largest code observed, plus one. The second is a fallback and warns,
        because a level missing from the training data cannot be recovered
        from X alone and would leave a bitstring too short at predict time.
        """
        matching_type = self.matching_type_

        if isinstance(matching_type, MixedTypes):
            categoricals = matching_type.categoricals
            mask = np.asarray(matching_type.mask, dtype=bool)
            columns = X[:, mask]
        elif isinstance(matching_type, GABIL):
            categoricals = matching_type
            columns = X
        else:
            return

        if categoricals.bits:
            return

        categoricals.bits = [np.zeros(n, dtype=bool) for n in self._level_counts(columns)]

    def _level_counts(self, columns: np.ndarray) -> list:
        """How many levels each categorical attribute has.

        An encoder supplied by the user is preferred, because it was fitted
        before the train-test split and therefore knows levels that this
        particular training set happens not to contain. Falling back to the
        largest observed code cannot recover those, so a rule would have a
        bitstring too short for a level that only appears at predict time.
        """
        encoder = getattr(self, "fitted_encoder_", None) or self.categorical_encoder

        if encoder is not None:
            categories = getattr(encoder, "categories_", None)
            if categories is not None:
                return [len(c) for c in categories]

            warnings.warn(
                "categorical_encoder has no categories_ attribute, so it "
                "appears not to be fitted. Falling back to the observed "
                "maximum, which cannot see levels absent from the training "
                "data."
            )

        warnings.warn(
            "Deriving the number of levels from the observed maximum. A "
            "level that does not appear in the training data cannot be "
            "detected this way and will raise at predict time. Pass a "
            "fitted encoder as categorical_encoder to avoid this."
        )

        return [int(columns[:, i].max()) + 1 for i in range(columns.shape[1])]    

    def _init_matching_type(self):
        for key, value in self.rule_discovery_.get_params().items():
            if "matching_type" in key:
                self.rule_discovery_.set_params(**{key: self.matching_type_})

    def _cleanup(self):
        """
        Optional cleanup of unused rules and components after fitting.
        Can be used to reduce size (e.g., for pickling) if only the final model is relevant.
        Note that all information about the fitting process itself is removed.
        """
        self.pool_ = self.elitist_.subpopulation

        self.elitist_.genome = np.ones(len(self.pool_), dtype="bool")
        self.elitist_.pool = self.pool_

        del self.rule_discovery_
        del self.solution_composition_

    def _more_tags(self):
        """
        For scikit-learn compatability, as it shows R^2 < 0.5 on certain data sets.
        Needed so that the model passes `check_estimator()`.
        """
        return {
            "poor_score": True,
        }
