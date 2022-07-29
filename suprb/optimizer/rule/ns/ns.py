import math

import numpy as np
from scipy.spatial.distance import hamming

from suprb.rule import Rule, RuleInit
from suprb.rule.initialization import HalfnormInit
from suprb.utils import check_random_state
from .crossover import RuleCrossover, UniformCrossover
from .novelty_calculation import NoveltyCalculation, NoveltySearchRule, ProgressiveMinimalCriteria, NoveltyFitnessBiased, NovelityFitnessPareto
from .novelty_search_type import BasicNoveltySearchType, LocalCompetition, MinimalCriteria
from .archive import Archive, ArchiveNone, ArchiveNovel, ArchiveRandom
from suprb.optimizer.rule.mutation import Normal, RuleMutation
from suprb.optimizer.rule.selection import RuleSelection, Random
from .. import RuleAcceptance, RuleConstraint
from ..acceptance import Variance
from ..base import RuleGeneration
from ..constraint import CombinedConstraint, MinRange, Clip
from ..origin import RuleOriginGeneration, UniformSamplesOrigin, SquaredError


class NoveltySearch(RuleGeneration):
    """ NoveltySearch Algorithm

        Parameters
        ----------
        n_iter: int
            Iterations to evolve rules. Must be greater than zero.
        mu: int
            The amount of offspring from each population that get selected for the next generation.
        lmbda: int
            Each generation lambda children will be generated.
        random_state : int, RandomState instance or None, default=None
            Pass an int for reproducible results across multiple function calls.
        n_jobs: int
            The number of threads / processes the optimization uses. Currently not used for this optimizer.
        n_elitists: int
            The number of parents that get added to the population each generation
        origin_generation: RuleOriginGeneration
            The selection process which decides on the next initial points.
        init: RuleInit
            A method to init rules. The init must always match at least one
            example but ideally should already match more than one,
            e.g. HalfnormInit, whereas NormalInit would not work consistently.
        crossover: RuleCrossover
        mutation: RuleMutation
        selection: RuleSelection
        acceptance: RuleAcceptance
        constraint: RuleConstraint
        novelty_calculation: NoveltyCalculation
            Class to calculate the novelty score based on NoveltySearchType, Archive and k_neigbor
        """

    last_iter_inner: bool

    def __init__(self,
                 n_iter: int = 10,
                 mu: int = 16,
                 lmbda: int = 160,
                 roh: int = 10,
                 random_state: int = None,
                 n_jobs: int = 1,
                 n_elitists=10,

                 origin_generation: RuleOriginGeneration = SquaredError(),
                 init: RuleInit = HalfnormInit(),
                 crossover: RuleCrossover = UniformCrossover(),
                 mutation: RuleMutation = Normal(sigma=0.1),
                 selection: RuleSelection = Random(),
                 acceptance: RuleAcceptance = Variance(),
                 constraint: RuleConstraint = CombinedConstraint(MinRange(),
                                                                 Clip()),
                 novelty_calculation: NoveltyCalculation = NoveltyCalculation(novelty_search_type=BasicNoveltySearchType(),
                                                                              archive=ArchiveNovel(),
                                                                              k_neighbor=15)):
        super().__init__(
            n_iter=n_iter,
            origin_generation=origin_generation,
            init=init,
            acceptance=acceptance,
            constraint=constraint,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.n_iter = n_iter
        self.mu = mu
        self.lmbda = lmbda
        self.roh = roh
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.n_elitists = n_elitists
        self.novelty_calculation = novelty_calculation

    def optimize(self, X: np.ndarray, y: np.ndarray, n_rules: int) -> list[Rule]:
        """ Validation of the parameters and checking the random_state.
            Then _optimize is called, where the Novelty Search algorithm is implemented.

            Return: A list of filtered Rules.
        """
        self._validate_params(n_rules)
        self.random_state_ = check_random_state(self.random_state)
        self.novelty_calculation.archive.set_random_state(self.random_state_)
        self.novelty_calculation.archive.set_archive(self.pool_)

        rules = self._optimize(X=X, y=y, n_rules=n_rules)

        return self._filter_invalid_rules(X=X, y=y, rules=rules)

    def _validate_params(self, n_rules: int):
        if n_rules > (self.mu + self.lmbda):
            raise ValueError(f"n_rules={n_rules} must be less or equal to mu+lambda={self.mu+self.lmbda}.")

    def _optimize(self, X: np.ndarray, y: np.ndarray, n_rules: int) -> list[Rule]:
        """ Steps of the novelty Search Algorithm containing:

            1. Initialisation of the Population.
            2. Loop of Selection, Crossover and Mutation and replacing the current Generation based on novelty.

            Return: n_rules generated by the algorithm
        """
        population = self._init_population(X, y)

        for i in range(self.n_iter):
            parents = self._select_shuffled_parents(population)
            children = self._crossover_and_mutate(X, y, parents)

            valid_children = list(filter(lambda rule: rule.is_fitted_ and rule.experience_ > 0, children))

            if valid_children:
                ns_children = self.novelty_calculation(rules=children)
                ns_parents = self.novelty_calculation(rules=parents)

                self.novelty_calculation.archive.extend_archive(ns_children, self.roh)

                best_children = self._get_n_best_rules(ns_children, self.mu)
                best_parents = self._get_n_best_rules(ns_parents, self.n_elitists)

                population = best_children + best_parents

        return [ns_rule for ns_rule in self.novelty_calculation.archive.get_archive() + best_children][:n_rules]

    def _init_population(self, X: np.ndarray, y: np.ndarray) -> list[Rule]:
        population = []
        half_mu = int(math.ceil(self.mu / 2))

        if len(self.pool_) < half_mu:
            population = self.pool_
            n_rules = self.mu - len(self.pool_)
        else:
            population = self.random_state_.choice(self.pool_, size=half_mu, replace=False).tolist()
            n_rules = half_mu

        origins = self.origin_generation(n_rules=n_rules, X=X, y=y, pool=self.pool_,
                                         elitist=self.elitist_, random_state=self.random_state_)
        for origin in origins:
            initialized_rules = self.init(mean=origin, random_state=self.random_state_)
            constrained_rules = self.constraint(initialized_rules)
            fitted_rules = constrained_rules.fit(X, y)
            population.append(fitted_rules)

        return population

    def _select_shuffled_parents(self, population: list[Rule]) -> list[Rule]:
        parents = self.selection(population, random_state=self.random_state_, size=self.lmbda)
        self.random_state_.shuffle(parents)

        return parents

    def _crossover_and_mutate(self, X: np.ndarray, y: np.ndarray, parents: list[Rule]) -> list[Rule]:
        children = []

        parent_combinations = zip(*[iter(parents)] * 2)
        for parent_A, parent_B in parent_combinations:
            children.extend(self.crossover(A=parent_A, B=parent_B, random_state=self.random_state_))

        children = [self.constraint(self.mutation(child, random_state=self.random_state_)).fit(X, y)
                    for child in children]

        return children

    def _get_n_best_rules(self, rules: list[NoveltySearchRule], n: int):
        sorted_rules = sorted(rules,
                              key=lambda ns_rule: (ns_rule.novelty_score, ns_rule.rule.experience_),
                              reverse=True)
        return [sorted_rule.rule for sorted_rule in sorted_rules][:n]
