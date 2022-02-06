import numpy as np
from scipy.spatial.distance import hamming

from suprb2.rule import Rule, RuleInit
from suprb2.rule.initialization import HalfnormInit
from .crossover import RuleCrossover, UniformCrossover
from .mutation import RuleMutation, Normal
from .selection import RuleSelection, Random
from .. import RuleAcceptance, RuleConstraint
from ..acceptance import Variance
from ..base import MultiRuleGeneration
from ..constraint import CombinedConstraint, MinRange, Clip
from ..origin import RuleOriginGeneration, UniformSamplesOrigin


class NoveltySearch(MultiRuleGeneration):

    def __init__(self,
                 n_iter: int = None,
                 lmbda: int = 160,       # has to be even
                 mu: int = 16,
                 origin_generation: RuleOriginGeneration = UniformSamplesOrigin(),
                 init: RuleInit = HalfnormInit(),
                 crossover: RuleCrossover = UniformCrossover(),
                 mutation: RuleMutation = Normal(sigma=0.1),
                 selection: RuleSelection = Random(),
                 acceptance: RuleAcceptance = Variance(),
                 constraint: RuleConstraint = CombinedConstraint(MinRange(), Clip()),
                 random_state: int = None,
                 n_jobs: int = 1,
                 minimal_criteria: bool = False,
                 threshold_fitness: float = None,
                 threshold_error: float = None,
                 threshold_amount_matched: int = None,
                 local_competition: bool = False,
                 archive: str = 'novelty'               # novelty, random or no
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

        self.lmbda = lmbda
        self.mu = mu
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.iterations = n_iter
        self.first_iter = True

        self.minimal_criteria = minimal_criteria
        self.threshold_fitness = threshold_fitness
        self.threshold_error = threshold_error
        self.threshold_amount_matched = threshold_amount_matched

        self.local_competition = local_competition

        self.archive = archive

    def _optimize(self, X: np.ndarray, y: np.ndarray, n_rules: int) -> list[Rule, float]:
        population = self._init_population(X, y)

        # main loop
        for i in range(self.iterations):

            # select lambda parents from population for crossover
            parents = self.selection(population, size=self.lmbda, random_state=self.random_state_)

            # from parents generate children through crossover and mutation
            children = []
            parents_iter = iter(parents)
            for parent in parents_iter:
                children.extend(self.crossover(A=parent, B=next(parents_iter), random_state=self.random_state_))

            children = [self.constraint(self.mutation(child, random_state=self.random_state_))
                            .fit(X, y) for child in children]

            # fill population for new iteration with 6/7 best children and 1/7 elitists
            population = self._new_population(X, y, children, parents)

            # at the end of first iteration change behaviour to normal
            if self.first_iter and i == self.iterations - 1 and self.archive != 'no':
                self.first_iter = False

        return population

    def _calculate_novelty_score(self, X: np.ndarray, y: np.ndarray, rules: list[Rule], archive: list[Rule], k: int) -> \
            list[Rule, float]:

        rules_with_novelty_score = []

        # filter rules for minimal criteria
        if self.minimal_criteria and not self.first_iter:
            rules = self._filter_for_minimal_criteria(rules)

        valid_archive = list(filter(lambda r: r.is_fitted_ and r.experience_ > 0, archive))

        # main loop with option for local competition
        for rule in rules:
            rules_w_distances = sorted([(B, hamming(rule.match_, B.match_)) for B in archive], key=lambda a: a[1])

            distances = [a[1] for a in rules_w_distances]
            novelty_score = sum(distances[:k]) / len(distances[:k])

            # no need to enter if there is no valid_archive
            if self.local_competition and valid_archive:
                count_worse = 0
                for x, _ in rules_w_distances[:int(round(len(rules_w_distances) / 20))]:
                    if x.fitness_ < rule.fitness_:
                        count_worse += 1
                local_score = count_worse / len(rules_w_distances[:int(round(len(rules_w_distances) / 20))])
                novelty_score = novelty_score + local_score

            # matched_data_count is the secondary key which decides what rule is better if novelty is the same
            matched_data_count = np.count_nonzero(rule.match_)

            rules_with_novelty_score.append((rule, novelty_score, matched_data_count))

        return rules_with_novelty_score

    def _init_population(self, X: np.ndarray, y: np.ndarray):
        population = []
        if len(self.pool_) < int(self.mu / 2):
            origins = self.origin_generation(n_rules=(self.mu - len(self.pool_)), X=X, pool=self.pool_,
                                             elitist=self.elitist_, random_state=self.random_state_)
            for origin in origins:
                population.append(self.constraint(self.init(mean=origin, random_state=self.random_state_)).fit(X, y))
                population.extend(self.pool_)
        else:
            origins = self.origin_generation(n_rules=int(self.mu / 2), X=X, pool=self.pool_, elitist=self.elitist_,
                                             random_state=self.random_state_)
            for origin in origins:
                population.append(self.constraint(self.init(mean=origin, random_state=self.random_state_)).fit(X, y))
            population.extend(self.random_state_.choice(self.pool_, size=int(self.mu / 2)))
        return population

    def _new_population(self, X: np.ndarray, y: np.ndarray, children: list[Rule], parents: list[Rule]):
        population = []
        if self.first_iter:
            children_w_ns = self._calculate_novelty_score(X, y, rules=children, archive=children, k=15)
            children_w_ns = sorted(children_w_ns, key=lambda x: (x[1], x[2]), reverse=True)
            population.extend([x[0] for x in children_w_ns][:int(round(self.mu * 6 / 7))])

            parents_w_ns = self._calculate_novelty_score(X, y, rules=parents, archive=parents, k=15)
            parents_w_ns = sorted(parents_w_ns, key=lambda x: (x[1], x[2]), reverse=True)
            population.extend([x[0] for x in parents_w_ns][:int(round(self.mu * 1 / 7))])
        else:
            children_w_ns = self._calculate_novelty_score(X, y, rules=children, archive=self.pool_, k=15)
            children_w_ns = sorted(children_w_ns, key=lambda x: (x[1], x[2]), reverse=True)
            population.extend([x[0] for x in children_w_ns][:int(round(self.mu * 6 / 7))])

            parents_w_ns = self._calculate_novelty_score(X, y, rules=parents, archive=self.pool_, k=15)
            parents_w_ns = sorted(parents_w_ns, key=lambda x: (x[1], x[2]), reverse=True)
            population.extend([x[0] for x in parents_w_ns][:int(round(self.mu * 1 / 7))])
        return population

    def _filter_for_minimal_criteria(self, rules: list[Rule]):
        if self.threshold_error:
            rules = [rule for rule in rules if rule.error_ > self.threshold_error]
        if self.threshold_fitness:
            rules = [rule for rule in rules if rule.fitness_ > self.threshold_fitness]
        if self.threshold_amount_matched:
            rules = [rule for rule in rules if np.count_nonzero(rule.match_) > self.threshold_amount_matched]
        return rules
