import numpy as np
from scipy.spatial.distance import hamming

from suprb2.rule import Rule, RuleInit
from suprb2.rule.initialization import HalfnormInit
from .crossover import RuleCrossover, UniformCrossover
from suprb2.optimizer.rule.mutation import Normal
from suprb2.optimizer.rule.selection import RuleSelection, Random
from .. import RuleAcceptance, RuleConstraint
from ..acceptance import Variance
from ..base import MultiRuleGeneration
from ..constraint import CombinedConstraint, MinRange, Clip
from ..origin import RuleOriginGeneration, UniformSamplesOrigin


class NoveltySearch(MultiRuleGeneration):

    def __init__(self,
                 n_iter: int = 100,
                 mu: int = 16,

                 origin_generation: RuleOriginGeneration = UniformSamplesOrigin(),
                 init: RuleInit = HalfnormInit(),
                 crossover: RuleCrossover = UniformCrossover(),
                 sigma: float = 0.1,
                 selection: RuleSelection = Random(),
                 acceptance: RuleAcceptance = Variance(),
                 constraint: RuleConstraint = CombinedConstraint(MinRange(), Clip()),
                 random_state: int = None,
                 n_jobs: int = 1,

                 ns_type: str = 'NS',                           # NS, NSLC or MCNS

                 threshold_amount_matched: int = None,

                 archive: str = 'novelty',                      # novelty, random or none
                 fitness_novelty_combination: str = 'novelty'   # novelty, 50/50, 25/75, pmcns or pareto
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
        self.lmbda = 10 * mu
        self.crossover = crossover
        self.sigma = sigma
        self.mutation = Normal(sigma=self.sigma)
        self.selection = selection
        self.iterations = n_iter
        self.first_iter_outer = True
        self.last_iter_inner = False

        self.ns_type = ns_type

        # params for MCNS
        self.threshold_amount_matched = threshold_amount_matched

        self.archive = archive
        self.fitness_novelty_combination = fitness_novelty_combination

    def _optimize(self, X: np.ndarray, y: np.ndarray, n_rules: int) -> list[Rule]:
        population = self._init_population(X, y)
        self.last_iter_inner = False

        # main loop
        for i in range(self.iterations):

            if i == self.iterations - 1:
                self.last_iter_inner = True

            # select lambda parents from population for crossover
            parents = self.selection(population, random_state=self.random_state_, size=self.lmbda)

            # from parents generate children through crossover and mutation
            children = []
            for j in range(0, len(parents) - 1, 2):
                children.extend(self.crossover(A=parents[j], B=parents[j + 1], random_state=self.random_state_))
            children = [self.constraint(self.mutation(child, random_state=self.random_state_))
                            .fit(X, y) for child in children]

            # fill population for new iteration with 6/7 best children and 1/7 elitists
            population = self._new_population(children, parents)

            # at the end of first iteration change behaviour to normal
            if self.first_iter_outer and i == self.iterations - 1 and self.archive != 'none':
                self.first_iter_outer = False

        return population

    def _calculate_novelty_score(self, rules: list[Rule], archive: list[Rule], k: int) -> list[tuple[Rule, float, int]]:

        rules_with_novelty_score = []

        # filter rules for minimal criteria
        if self.ns_type == 'MCNS' and not self.first_iter_outer:
            rules = self._filter_for_minimal_criteria(rules)

        if self.fitness_novelty_combination == 'pmcns':
            rules = self._filter_for_progressive_minimal_criteria(rules)

        valid_archive = list(filter(lambda r: r.is_fitted_ and r.experience_ > 0, archive))

        # main loop with option for local competition
        for rule in rules:
            rules_w_distances = sorted([(B, hamming(rule.match_, B.match_)) for B in archive], key=lambda a: a[1])

            distances = [a[1] for a in rules_w_distances]
            novelty_score = sum(distances[:k]) / len(distances[:k])

            # no need to enter if there is no valid_archive
            if self.ns_type == 'NSLC' and valid_archive:
                count_worse = 0
                for x, _ in rules_w_distances[:int(round(len(rules_w_distances) / 20))]:
                    if x.fitness_ < rule.fitness_:
                        count_worse += 1
                local_score = count_worse / len(rules_w_distances[:int(round(len(rules_w_distances) / 20))])
                novelty_score = novelty_score + local_score

            # matched_data_count is the secondary key which decides what rule is better if novelty is the same
            matched_data_count = np.count_nonzero(rule.match_)

            # linear combination of fitness and novelty
            if self.fitness_novelty_combination == '50/50':
                novelty_score = novelty_score + (rule.fitness_ / 100)
            elif self.fitness_novelty_combination == '25/75':
                novelty_score = novelty_score + (rule.fitness_ / 200)

            rules_with_novelty_score.append((rule, novelty_score, matched_data_count))

        if self.fitness_novelty_combination == 'pareto' and not self.first_iter_outer:
            rules_with_novelty_score = sorted(rules_with_novelty_score, key=lambda a: (a[1], a[0].fitness_), reverse=True)
            p_front = [rules_with_novelty_score[0]]

            for r in rules_with_novelty_score[1:]:
                if r[0].fitness_ >= p_front[-1][0].fitness_:
                    p_front.append(r)

            rules_with_novelty_score = p_front

        return rules_with_novelty_score

    def _init_population(self, X: np.ndarray, y: np.ndarray) -> list[Rule]:
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

    def _new_population(self, children: list[Rule], parents: list[Rule]) -> list[Rule]:
        population = []

        if not self.first_iter_outer:
            archive_children = self.pool_
            archive_parents = self.pool_
        else:
            archive_children = children
            archive_parents = parents

        children_w_ns = self._calculate_novelty_score(rules=children, archive=archive_children, k=15)
        parents_w_ns = self._calculate_novelty_score(rules=parents, archive=archive_parents, k=15)

        if self.last_iter_inner and self.archive == 'random':
            self.random_state_.shuffle(children_w_ns)
            self.random_state_.shuffle(parents_w_ns)
        else:
            children_w_ns = sorted(children_w_ns, key=lambda x: (x[1], x[2]), reverse=True)
            parents_w_ns = sorted(parents_w_ns, key=lambda x: (x[1], x[2]), reverse=True)

        population.extend([x[0] for x in children_w_ns][:int(round(self.mu * 6 / 7))])
        population.extend([x[0] for x in parents_w_ns][:int(round(self.mu * 1 / 7))])
        return population

    def _filter_for_minimal_criteria(self, rules: list[Rule]) -> list[Rule]:
        if self.threshold_amount_matched:
            rules = [rule for rule in rules if np.count_nonzero(rule.match_) > self.threshold_amount_matched]
        return rules

    def _filter_for_progressive_minimal_criteria(self, rules: list[Rule]) -> list[Rule]:
        threshold_fitness = np.median([rule.fitness_ for rule in rules])
        rules = [rule for rule in rules if rule.fitness_ >= threshold_fitness]
        return rules
