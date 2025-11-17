import cProfile
import pstats

import numpy as np
from typing import Optional, List, Callable, Tuple
from joblib import Parallel, delayed

from suprb.rule import Rule, RuleInit
from suprb.rule.initialization import MeanInit
from suprb.utils import RandomState
from ..base import MultiRuleDiscovery
from ..origin import Matching, SquaredError, RuleOriginGeneration
from ..mutation import RuleMutation, Normal
from ..constraint import CombinedConstraint, MinRange, Clip
from ..acceptance import Variance
from .. import RuleAcceptance, RuleConstraint
from .nsga2_helpers import visualize_pareto_front

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance, FunctionalDiversity


class NSGA2(MultiRuleDiscovery):
    """
    Adapted from: A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II
    by Kalyanmoy Deb et al..

    This class implements a multi-objective evolutionary algorithm for
    rule discovery. It uses pymoo's fast nondominated sorting and crowding
    distance operators, and serves as a base class for multi-objective
    optimization in MOO-RD.

    Parameters
    ----------
    n_iter : int
        Number of evolutionary iterations.
    mu : int
        Population size
    lmbda : int
        Number of children sampled each generation.
    origin_generation : RuleOriginGeneration
    init : RuleInit
    mutation : RuleMutation
    constraint : RuleConstraint
    acceptance : RuleAcceptance
    random_state : int or None
    n_jobs : int
    fitness_objs : list, optional
        List of fitness objectives
        Defaults to [rule.error_, -rule.volume_].
    fitness_objs_labels : list of str, optional
        Names corresponding to the objective functions.
        Defaults to ["obj_0", "obj_1", ...].
    profile : bool, default=False
        If True, wraps the optimization loop in a profiler and prints stats.
    """
    def __init__(
            self,
            n_iter: int = 32,
            mu: int = 50,
            lmbda: int = 100,
            origin_generation: RuleOriginGeneration = SquaredError(),
            init: RuleInit = MeanInit(),
            mutation: RuleMutation = Normal(sigma=1.22),
            constraint: RuleConstraint = CombinedConstraint(MinRange(), Clip()),
            acceptance: RuleAcceptance = Variance(),
            random_state: int = None,
            n_jobs: int = 1,
            fitness_objs: Optional[List[Callable[[Rule], float]]] = None,
            fitness_objs_labels: Optional[List[str]] = None,
            profile: bool = False,
    ):
        super().__init__(
            n_iter=n_iter,
            origin_generation=origin_generation,
            init=init,
            acceptance=acceptance,
            constraint=constraint,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.mu = mu
        self.lmbda = lmbda
        self.mutation = mutation
        self.constraint = constraint

        if fitness_objs is None:
            fitness_objs = [
                lambda r: r.error_,
                lambda r: -r.volume_,
            ]

        # Extra objectives in subclasses are added during runtime to avoid sklearn.clone errors.
        self.fitness_objs: Tuple[Callable[[Rule], float], ...] = tuple(fitness_objs)

        if fitness_objs_labels is None:
            fitness_objs_labels = [f"obj_{i}" for i in range(len(self.fitness_objs))]
        self.fitness_objs_labels: Tuple[str, ...] = tuple(fitness_objs_labels)

        self.profile = profile

    def _optimize(
            self,
            X: np.ndarray,
            y: np.ndarray,
            random_state: RandomState
    ) -> Optional[List[Rule]]:
        profiler = cProfile.Profile() if self.profile else None
        if profiler:
            profiler.enable()

        # Generate initial population.
        population = []
        origins = self.origin_generation(
            n_rules=self.mu,
            X=X,
            y=y,
            pool=self.pool_,
            elitist=self.elitist_,
            random_state=random_state,
        )

        population = Parallel(n_jobs=self.n_jobs)(
            delayed(self._init_valid_origin)(origin, X, y, random_state)
            for origin in origins
        )
        population = [p for p in population if p is not None]

        # Main loop.
        for _ in range(self.n_iter):
            # Select parents and generate children.
            parents = random_state.choice(population, size=self.lmbda, replace=True)
            children = Parallel(n_jobs=self.n_jobs)(
                delayed(self._generate_valid_child)(parent, X, y, random_state)
                for parent in parents
            )
            children = [c for c in children if c is not None]

            # Combine current population and children (mu + lmbda), then nondominated sorting into fronts.
            population_combined = population + children
            pareto_fronts = self._fast_nondominated_sort(population_combined)
            population = self._build_next_population(pareto_fronts) #asigns crowding distance, build next population of size mu.

        # The nondominated set.
        pareto_front = pareto_fronts[0] if pareto_fronts else []

        if profiler:
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats("cumtime")
            stats.print_stats(20)

        # Visualize the objective space distribution for the chosen fitness objectives. Is intended for the isolated experiments.
        # Comment out if not run in isolation or for more than one iteration.
        #visualize_pareto_front(self, pareto_front)

        return pareto_front

# ────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────
    def _fast_nondominated_sort(self, population: List[Rule]) -> List[List[Rule]]:
        """
        Sorts population into fronts using pymoo's fast nondominated sort.
        """
        if not population:
            return []
        objs = self._fitness_objs_runtime()
        obj_matrix = np.vstack(
            [[obj(rule) for obj in objs] for rule in population]
        )

        fronts = NonDominatedSorting().do(obj_matrix, only_non_dominated_front=False)
        return [[population[i] for i in front] for front in fronts]


    def _assign_crowding_distance(
            self,
            front: List[Rule],
            cd_func,
    ):
        """
        Calculate crowding distance using pymoo's crowding distance function.
        """
        if not front:
            return

        objs = self._fitness_objs_runtime()
        obj_matrix = np.vstack(
            [[obj(rule) for obj in objs] for rule in front]
        )

        crowding_distances = cd_func.do(obj_matrix)

        for rule, dist in zip(front, crowding_distances):
            rule.crowding_distance_ = dist


    def _init_valid_origin(
            self,
            origin,
            X,
            y,
            random_state,
    ):
        rule = self.constraint(self.init(mean=origin, random_state=random_state)).fit(X, y)
        return rule if rule.is_fitted_ and rule.experience_> 0 else None


    def _generate_valid_child(
        self,
        parent,
        X,
        y,
        random_state,
    ):
        child = self.constraint(self.mutation(parent, random_state)).fit(X,y)
        return child if child.is_fitted_ and child.experience_ > 0 else None


    def _build_next_population(
            self,
            pareto_fronts,
    ):
        """
        Add fronts starting from F_0 until population limit mu is reached.
        Last front is truncated if necessary and the population then is extended by the individuals with the highest crowding distance.
        """
        population_new = []

        cd_func = FunctionalDiversity(calc_crowding_distance, filter_out_duplicates=True)

        for front in pareto_fronts:
            self._assign_crowding_distance(front, cd_func)
            if len(population_new) + len(front) <= self.mu:
                population_new.extend(front)
            else:
                front_sorted = sorted(front, key=lambda rule: -rule.crowding_distance_)
                population_new.extend(front_sorted[:self.mu - len(population_new)])
                break

        return population_new


    def _fitness_objs_runtime(self) -> List[Callable[[Rule], float]]:
        """
        Used in the subclasses to add the secondary objective at runtime.
        If set otherwise, the objectives generate a sklearn error.
        """
        return list(self.fitness_objs)


    def _fitness_labels_runtime(self) -> List[str]:
        """
        Returns list of all fitness objectives used.
        """
        return list(self.fitness_objs_labels)
