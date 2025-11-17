import numpy as np
from typing import Optional, List, Callable
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from suprb.rule import Rule, RuleInit
from suprb.rule.initialization import MeanInit
from suprb.optimizer.rule.nsga2.nsga2 import NSGA2
from suprb.optimizer.rule.ns.novelty_calculation import NoveltyCalculation
from suprb.optimizer.rule.ns.archive import ArchiveNovel
from suprb.optimizer.rule.ns.novelty_search_type import NoveltySearchType
from suprb.utils import RandomState
from ..origin import Matching, SquaredError, RuleOriginGeneration
from ..mutation import RuleMutation, HalfnormIncrease
from ..constraint import CombinedConstraint, MinRange, Clip
from ..acceptance import Variance
from .. import RuleAcceptance, RuleConstraint
from .nsga2_helpers import visualize_pareto_front

import cProfile
import pstats



class NSGA2GlobalNovelty(NSGA2):
    """
    Adapted from: A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II by Kalyanmoy Deb et al.
    Uses Crowding Distance and NonDominatedSorting from pymoo package.
    Uses novelty as a fitness objective.
    Not optimized. Deprecated.
    """
    def __init__(
        self,
        n_iter: int = 32,
        mu: int = 50,
        lmbda: int = 100,
        origin_generation: RuleOriginGeneration = SquaredError(),
        init: RuleInit = MeanInit(),
        mutation: RuleMutation = HalfnormIncrease(sigma=1.22),
        constraint: RuleConstraint = CombinedConstraint(MinRange(), Clip()),
        acceptance: RuleAcceptance = Variance(),
        random_state: int = None,
        n_jobs: int = 1,
        fitness_objs: Optional[List[Callable[[Rule], float]]] = None,
        fitness_objs_labels: Optional[List[str]] = None,
        novelty_calc: NoveltyCalculation = NoveltyCalculation(
            novelty_search_type=NoveltySearchType(),
            archive=ArchiveNovel(),
            k_neighbor=15,
        ),
        profile: bool = False,
    ):
        super().__init__(
            n_iter=n_iter,
            mu=mu,
            lmbda=lmbda,
            origin_generation=origin_generation,
            init=init,
            mutation=mutation,
            constraint=constraint,
            acceptance=acceptance,
            random_state=random_state,
            n_jobs=n_jobs,
            fitness_objs=fitness_objs,
            fitness_objs_labels=fitness_objs_labels,
        )
        self.novelty_calc = novelty_calc

        self.fitness_objs = list(self.fitness_objs) + [lambda r: -r.novelty_score_]
        self.fitness_objs_labels = list(self.fitness_objs_labels) + ['-Novelty']    

    def _optimize(
            self,
            X: np.ndarray,
            y: np.ndarray,
            random_state: RandomState
    ) -> Optional[List[Rule]]:
        profiler = cProfile.Profile() if self.profile else None
        if profiler:
            profiler.enable()
            
        self.pool_.clear()

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

        for _ in range(self.n_iter):
            parents = random_state.choice(population, size=self.lmbda, replace=True)

            children = Parallel(n_jobs=self.n_jobs)(
                delayed(self._generate_valid_child)(parent, X, y, random_state)
                for parent in parents
            )

            children = [c for c in children if c is not None]

            population_combined = population + children

            self.pool_.extend(population_combined)
            self.novelty_calc.archive.archive = self.pool_
            _ = self.novelty_calc(self.pool_)

            pareto_fronts = self._fast_nondominated_sort(population_combined)
            population = self._build_next_population(pareto_fronts)

        pareto_front = pareto_fronts[0]
        if not pareto_front:
            return None

        if profiler:
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats("cumtime")
            stats.print_stats(20)

        visualize_pareto_front(self, pareto_front)

        return pareto_front
        