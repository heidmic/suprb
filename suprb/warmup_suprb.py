import numpy as np
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_array

from . import SupRB
from .solution import Solution
from .solution.mixing_model import ErrorExperienceHeuristic
from .solution.fitness import PseudoBIC
from .optimizer.rule.es import ES1xLambda
from .optimizer.solution.ga import GeneticAlgorithm
from .rule.matching import OrderedBound
from .logging import DefaultLogger
from .utils import check_random_state


class WarmupSupRB(SupRB):
    """
    SupRB variant that performs several Rule-Discovery-only warm-up cycles
    before the first Solution Composition.

    DEPRECATED. Functionality is already implemented in SupRB via n_initial_rules.
    """

    def __init__(
        self,
        rule_discovery=None,
        solution_composition=None,
        matching_type=None,
        n_iter=32,
        n_initial_rules=0,
        n_rules=4,
        random_state=None,
        verbose=1,
        logger=None,
        n_jobs=1,
        early_stopping_patience=-1,
        early_stopping_delta=0,
        # Warmup params
        warmup_strategy="fixed",
        warmup_rd_steps=0,               # fixed
        warmup_max_steps=16,             # auto
        warmup_pool_target=None,         # auto
        warmup_patience=3,               # auto
        warmup_delta=1,                  # auto
    ):
        super().__init__(
            rule_discovery=rule_discovery,
            solution_composition=solution_composition,
            matching_type=matching_type,
            n_iter=n_iter,
            n_initial_rules=n_initial_rules,
            n_rules=n_rules,
            random_state=random_state,
            verbose=verbose,
            logger=logger,
            n_jobs=n_jobs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_delta=early_stopping_delta,
        )
        self.warmup_strategy = warmup_strategy
        self.warmup_rd_steps = warmup_rd_steps
        self.warmup_max_steps = warmup_max_steps
        self.warmup_pool_target = warmup_pool_target
        self.warmup_patience = warmup_patience
        self.warmup_delta = warmup_delta


    def fit(self, X, y, cleanup=False):
        self.early_stopping_counter_ = 0
        self.previous_fitness_ = 0
        self.is_error_ = False

        self.elitist_ = Solution([0, 0, 0], [0, 0, 0], ErrorExperienceHeuristic(), PseudoBIC())
        self.elitist_.fitness_ = 0
        self.elitist_.error_ = 99999
        self.elitist_.complexity_ = 99999

        X, y = check_X_y(X, y, dtype="float64", y_numeric=True)
        y = check_array(y, ensure_2d=False, dtype="float64")
        self.n_features_in_ = X.shape[1]

        self.random_state_ = check_random_state(self.random_state)
        total_pairs = (self.warmup_max_steps + self.n_iter) * 2
        seeds = np.random.SeedSequence(self.random_state).spawn(total_pairs)
        self.rule_discovery_seeds_ = seeds[::2]
        self.solution_composition_seeds_ = seeds[1::2]

        self.pool_ = []
        self._validate_rule_discovery(default=ES1xLambda())
        self._validate_solution_composition(default=GeneticAlgorithm())
        self._validate_matching_type(default=OrderedBound(np.array([])))
        self._validate_logger(default=DefaultLogger())
        self._propagate_component_parameters()
        self._init_bounds(X)
        self._init_matching_type()

        self.solution_composition_.pool_ = self.pool_
        self.rule_discovery_.pool_ = self.pool_

        self.solution_composition_.init.fitness.max_genome_length_ = (
            self.n_rules * (self.n_iter + self.warmup_max_steps) + self.n_initial_rules
        )

        self.logger_.log_init(X, y, self)

        if self.n_initial_rules > 0:
            if self._catch_errors(self._discover_rules, X, y, self.n_initial_rules):
                return self

        # Warmup
        warmup_done = 0
        patience_ctr = 0
        self.step_ = 0

        if self.warmup_strategy not in ("fixed", "auto"):
            raise ValueError("warmup_strategy must be 'fixed' or 'auto'.")

        def one_rd_cycle():
            before = len(self.pool_)
            if self._catch_errors(self._discover_rules, X, y, self.n_rules):
                return True, 0
            after = len(self.pool_)
            growth = after - before
            if self.verbose >= 2:
                print(f"[Warm-up RD] Added {growth} rules (pool={after})")
            return False, growth

        if self.warmup_strategy == "fixed":
            target = max(0, int(self.warmup_rd_steps))
            for _ in range(target):
                err, _growth = one_rd_cycle()
                if err:
                    return self
                warmup_done += 1
                self.step_ += 1
        else:  # auto
            for _ in range(max(0, int(self.warmup_max_steps))):
                err, growth = one_rd_cycle()
                if err:
                    return self
                warmup_done += 1
                self.step_ += 1

                if self.warmup_pool_target is not None and len(self.pool_) >= int(self.warmup_pool_target):
                    break
                if growth < int(self.warmup_delta):
                    patience_ctr += 1
                    if patience_ctr >= int(self.warmup_patience):
                        break
                else:
                    patience_ctr = 0

        self.solution_composition_.init.fitness.max_genome_length_ = (
            self.n_rules * (self.n_iter + warmup_done) + self.n_initial_rules
        )

        # Main loop
        end_step = self.step_ + self.n_iter
        while self.step_ < end_step:
            if self._catch_errors(self._discover_rules, X, y, self.n_rules):
                return self
            if self._catch_errors(self._compose_solution, X, y, False):
                return self

            self.logger_.log_iteration(X, y, self, iteration=self.step_)
            if self.check_early_stopping():
                break

            self.previous_fitness_ = self.solution_composition_.elitist().fitness_
            self.step_ += 1

        self.elitist_ = self.solution_composition_.elitist().clone()
        self.is_fitted_ = True
        self.logger_.log_final(X, y, self)

        if cleanup:
            self._cleanup()

        return self

    def _discover_rules(self, X, y, n_rules):
        self._log_to_stdout(f"Generating {n_rules} rules", priority=4)

        elit = None
        try:
            elit = self.solution_composition_.elitist()
        except Exception:
            pass
        if elit is None:
            elit = self.elitist_

        self.rule_discovery_.elitist_ = elit
        self.rule_discovery_.random_state = self.rule_discovery_seeds_[self.step_]
        new_rules = self.rule_discovery_.optimize(X, y, n_rules=n_rules)
        self.pool_.extend(new_rules)

