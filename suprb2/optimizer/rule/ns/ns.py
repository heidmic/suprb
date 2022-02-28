import math

import numpy as np
from scipy.spatial.distance import hamming

from suprb2.rule import Rule, RuleInit
from suprb2.rule.initialization import HalfnormInit
from suprb2.utils import check_random_state
from .crossover import RuleCrossover, UniformCrossover
from suprb2.optimizer.rule.mutation import Normal, RuleMutation
from suprb2.optimizer.rule.selection import RuleSelection, Random
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
            Iterations to evolve rules.
        mu: int
            The amount of offspring from each population that get selected for the next generation.
        lm_ratio: int
            The ratio of lambda and mu. Each generation lambda=lm_ration*mu children will be generated but only mu will
            survive.
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
        random_state : int, RandomState instance or None, default=None
            Pass an int for reproducible results across multiple function calls.
        n_jobs: int
            The number of threads / processes the optimization uses. Currently not used for this optimizer.
        ns_type: str
            The type of Novelty Search to be used. Can be 'NS' for standard Novelty Search, 'NSLC' for Novelty Search
            with Local Competition or 'MCNS' for Minimal Criteria Novelty Search.
        MCNS_threshold_matched: int
            The amount of samples a rule must match when using MCNS. Otherwise this parameter has no effect.
        archive: str
            The type of archive to be used by the algorithm. Can be 'novelty' for an archive with the most novel
            individuals of each call of _optimize, 'random' where individuals will be chosen randomly from the final
            population or  'none' where there is no archive and only the current generation is used to calculate
            novelty.
        novelty_fitness_combination: str
            The type of novelty-fitness combination. Can be 'novelty' where only novelty affects the score of an
            individual, '50/50' or '75/25' which gives a linear combination of novelty and fitness with the according
            weights, 'pmcns' which stands for progressive MCNS and only allows the best 50% of the current population
            in fitness to be considered for the final offspring of a generation or 'pareto' which calculates the pareto
            front of each generation and assigns each individual a score through that.
        """

    last_iter_inner: bool

    def __init__(self,
                 n_iter: int = 10,
                 mu: int = 16,
                 lm_ratio: int = 10,

                 origin_generation: RuleOriginGeneration = SquaredError(),
                 init: RuleInit = HalfnormInit(),
                 crossover: RuleCrossover = UniformCrossover(),
                 mutation: RuleMutation = Normal(sigma=0.1),
                 selection: RuleSelection = Random(),
                 acceptance: RuleAcceptance = Variance(),
                 constraint: RuleConstraint = CombinedConstraint(MinRange(), Clip()),
                 random_state: int = None,
                 n_jobs: int = 1,

                 ns_type: str = 'MCNS',  # NS, NSLC or MCNS

                 MCNS_threshold_matched: int = 30,

                 archive: str = 'novelty',  # novelty, random or none
                 novelty_fitness_combination: str = 'novelty'  # novelty, 50/50, 75/25, pmcns or pareto
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
        self.lm_ratio = lm_ratio
        self.lmbda = self.lm_ratio * mu
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.n_iter = n_iter

        self.ns_type = ns_type

        # parameter only for MCNS
        self.MCNS_threshold_matched = MCNS_threshold_matched

        self.archive = archive
        self.novelty_fitness_combination = novelty_fitness_combination

    def optimize(self, X: np.ndarray, y: np.ndarray, n_rules: int) -> list[Rule]:
        """ Validation of the parameters and checking the random_state.
            Then _optimize is called, where the Novelty Search algorithm is implemented.

            Return: A list of filtered Rules.
        """
        self._validate_params(n_rules)

        self.random_state_ = check_random_state(self.random_state)

        rules = self._optimize(X=X, y=y, n_rules=n_rules)

        return self._filter_invalid_rules(X=X, y=y, rules=rules)

    def _optimize(self, X: np.ndarray, y: np.ndarray, n_rules: int) -> list[Rule]:
        """ Steps of the novelty Search Algorithm containing:

            1. Initialisation of the Population.
            2. Loop of Selection, Crossover and Mutation and replacing the current Generation based on novelty.

            Return: n_rules generated by the algorithm
        """
        population = self._init_population(X, y)
        self.last_iter_inner = False

        # main loop
        for i in range(self.n_iter):

            if i == self.n_iter - 1:
                self.last_iter_inner = True

            # select lambda parents from population for crossover
            parents = self.selection(population, random_state=self.random_state_, size=self.lmbda)

            self.random_state_.shuffle(parents)

            # from parents generate children through crossover and mutation
            children = []
            for j in range(0, len(parents) - 1, 2):
                children.extend(self.crossover(A=parents[j], B=parents[j + 1], random_state=self.random_state_))
            children = [self.constraint(self.mutation(child, random_state=self.random_state_))
                            .fit(X, y) for child in children]

            # filter children
            valid_children = list(filter(lambda rule: rule.is_fitted_ and rule.experience_ > 0, children))

            # fill population for new iteration with mu children and n_rules elitists except for last iteration
            # where children and parents will be combined and n_rules are chosen
            population = self._new_population(valid_children, parents, n_rules)

        return population

    def _calculate_novelty_score(self, rules: list[Rule], archive: list[Rule], k: int) -> list[tuple[Rule, float, int]]:
        """ Calculation of the Novelty Score based on the hamming distance.

            Takes every rule and calculates the hamming distance to each rule contained in the archive and averages the
            distances of the k-nearest neighbors to get the novelty score.

            Return: A list of tuples containing each rule with its corresponding novelty score and also the amount of
            matched data points, which is used as the tiebreaker in case of equal novelty score.
        """
        rules_with_novelty_score = []

        # filter rules for minimal criteria
        if self.ns_type == 'MCNS' and self.pool_:
            rules = self._filter_for_minimal_criteria(rules)

        if self.novelty_fitness_combination == 'pmcns':
            rules = self._filter_for_progressive_minimal_criteria(rules)

        valid_archive = list(filter(lambda r: r.is_fitted_ and r.experience_ > 0, archive))

        # main loop with option for local competition
        for rule in rules:
            rules_w_distances = sorted([(B, hamming(rule.match_, B.match_)) for B in archive], key=lambda a: a[1])

            distances = [a[1] for a in rules_w_distances]
            novelty_score = sum(distances[:k]) / len(distances[:k])

            # no need to enter if there is no valid_archive
            if self.ns_type == 'NSLC' and valid_archive:
                novelty_score = novelty_score + self._get_local_score(rule, rules_w_distances)

            # matched_data_count is the secondary key which decides what rule is better if novelty is the same
            matched_data_count = np.count_nonzero(rule.match_)

            # linear combination of fitness and novelty
            # any changes to fitness scaling would need to be also applied here
            scaled_fitness = rule.fitness_ / 100
            if self.novelty_fitness_combination == '50/50':
                novelty_score = 0.5 * novelty_score + 0.5 * scaled_fitness
            elif self.novelty_fitness_combination == '75/25':
                novelty_score = 0.75 * novelty_score + 0.25 * scaled_fitness

            rules_with_novelty_score.append((rule, novelty_score, matched_data_count))

        if self.novelty_fitness_combination == 'pareto' and self.pool_:
            rules_with_novelty_score = self._get_pareto_front(rules_with_novelty_score)

        return rules_with_novelty_score

    def _init_population(self, X: np.ndarray, y: np.ndarray) -> list[Rule]:
        population = []

        if len(self.pool_) < int(math.ceil(self.mu / 2)):
            n_rules = self.mu - len(self.pool_)
        else:
            n_rules = int(math.ceil(self.mu / 2))

        origins = self.origin_generation(n_rules=n_rules, X=X, y=y, pool=self.pool_,
                                         elitist=self.elitist_, random_state=self.random_state_)

        for origin in origins:
            population.append(self.constraint(self.init(mean=origin, random_state=self.random_state_)).fit(X, y))

        if len(self.pool_) < int(math.ceil(self.mu / 2)):
            population.extend(self.pool_)
        else:
            population.extend(self.random_state_.choice(self.pool_, size=int(self.mu / 2), replace=False))

        return population

    def _new_population(self, children: list[Rule], parents: list[Rule], n_rules: int) -> list[Rule]:
        population = []

        if self.pool_ and self.archive != 'none':
            archive_children = self.pool_
            archive_parents = self.pool_
        else:
            archive_children = children
            archive_parents = parents

        children_w_ns = self._calculate_novelty_score(rules=children, archive=archive_children, k=15)
        parents_w_ns = self._calculate_novelty_score(rules=parents, archive=archive_parents, k=15)

        if self.last_iter_inner:
            population = children_w_ns + parents_w_ns
            if self.archive == "random":
                self.random_state_.shuffle(population)
            else:
                population = sorted(population, key=lambda x: (x[1], x[2]), reverse=True)
            return [x[0] for x in population][:n_rules]
        else:
            children_w_ns = sorted(children_w_ns, key=lambda x: (x[1], x[2]), reverse=True)
            parents_w_ns = sorted(parents_w_ns, key=lambda x: (x[1], x[2]), reverse=True)
            population.extend([x[0] for x in children_w_ns][:self.mu])
            population.extend([x[0] for x in parents_w_ns][:n_rules])
            return population

    def _filter_for_minimal_criteria(self, rules: list[Rule]) -> list[Rule]:
        # calculate the 25th percentile value and if it's lower than MNCS_threshold_matched use it instead to filter
        # a maximum of 25% of the population (to prevent empty populations)
        maximum_threshold = np.percentile([np.count_nonzero(rule.match_) for rule in rules], 25)
        if maximum_threshold < self.MCNS_threshold_matched:
            rules = [rule for rule in rules if np.count_nonzero(rule.match_) >= maximum_threshold]
        else:
            rules = [rule for rule in rules if np.count_nonzero(rule.match_) >= self.MCNS_threshold_matched]
        return rules

    def _filter_for_progressive_minimal_criteria(self, rules: list[Rule]) -> list[Rule]:
        threshold_fitness = np.median([rule.fitness_ for rule in rules])
        rules = [rule for rule in rules if rule.fitness_ >= threshold_fitness]
        return rules

    def _get_local_score(self, rule: Rule, rules_w_distances: list[tuple[Rule, float]]) -> float:
        count_worse = 0
        for x, _ in rules_w_distances[:15]:
            if x.fitness_ < rule.fitness_:
                count_worse += 1
        local_score = count_worse / len(rules_w_distances[:15])
        return local_score

    def _get_pareto_front(self, rules_with_novelty_score: list[tuple[Rule, float, int]]) -> list[tuple[Rule, float,
                                                                                                       int]]:
        rules_with_novelty_score = sorted(rules_with_novelty_score, key=lambda a: (a[1], a[0].fitness_), reverse=True)
        p_front = [rules_with_novelty_score[0]]

        for r in rules_with_novelty_score[1:]:
            if r[0].fitness_ >= p_front[-1][0].fitness_:
                p_front.append(r)

        return p_front

    def _validate_params(self, n_rules: int):
        if self.ns_type not in ["NS", "NSLC", "MCNS"]:
            raise ValueError(f"{self.ns_type} is no valid NS-Type.")
        if self.archive not in ["novelty", "random", "none"]:
            raise ValueError(f"{self.archive} is no valid Archive-Type.")
        if self.novelty_fitness_combination not in ["novelty", "50/50", "75/25", "pmcns", "pareto"]:
            raise ValueError(f"{self.novelty_fitness_combination} is no valid Fitness-Novelty Combination.")
        if n_rules > (self.mu + self.lmbda):
            raise ValueError(f"n_rules={n_rules} must be less or equal to mu+lambda={self.mu+self.lmbda}.")
