import warnings
from typing import Optional, Iterator

import numpy as np
from scipy.spatial.distance import jaccard

from suprb2.rule import Rule, RuleInit
from suprb2.rule.initialization import HalfnormInit
from suprb2.utils import RandomState
from .crossover import RuleCrossover, UniformCrossover
from .mutation import RuleMutation, Normal, Halfnorm
from .selection import RuleSelection, Random
from .. import RuleAcceptance, RuleConstraint
from ..acceptance import Variance
from ..base import MultiRuleGeneration
from ..constraint import CombinedConstraint, MinRange, Clip
from ..origin import Matching, RuleOriginGeneration, UniformSamplesOrigin


class NoveltySearch(MultiRuleGeneration):

    def __init__(self,
                 n_iter: int = 10,
                 lmbda: int = 70,
                 mu: int = 7,
                 origin_generation: RuleOriginGeneration = UniformSamplesOrigin(),
                 init: RuleInit = HalfnormInit(),
                 crossover: RuleCrossover = UniformCrossover(),
                 mutation: RuleMutation = Normal(),
                 selection: RuleSelection = Random(),
                 acceptance: RuleAcceptance = Variance(),
                 constraint: RuleConstraint = CombinedConstraint(MinRange(), Clip()),
                 random_state: int = None,
                 n_jobs: int = 1,
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
        self.iterations = 10
        self.first_iter = True

    def _optimize(self, X: np.ndarray, y: np.ndarray, n_rules: int) -> list[Rule, float]:

        population = []

        # if archive has less than mu/2 rules generate (mu-len(archive)) new ones and take whole archive into
        # population; otherwise take mu/2 randomly from archive and generate mu/2 new rules
        if len(self.pool_) < int(self.mu / 2):

            origins = self.origin_generation(n_rules=(self.mu - len(self.pool_)), X=X, pool=self.pool_,
                                             elitist=self.elitist_,
                                             random_state=self.random_state_)

            for origin in origins:
                population.append(self.constraint(self.init(mean=origin, random_state=self.random_state_)).fit(X, y))
                population.extend(self.pool_)

        else:

            origins = self.origin_generation(n_rules=int(self.mu / 2), X=X, pool=self.pool_, elitist=self.elitist_,
                                             random_state=self.random_state_)

            for origin in origins:
                population.append(self.constraint(self.init(mean=origin, random_state=self.random_state_)).fit(X, y))

            population.extend(self.random_state_.choice(self.pool_, size=int(self.mu / 2)))


        # main loop
        for i in range(self.iterations):

            # select lambda parents from population for crossover
            parents = self.selection(population, size=self.lmbda, random_state=self.random_state_)
            parents_iterator = iter(parents)

            # from parents generate children through crossover and mutation
            children = []
            for parent in parents_iterator:
                children.extend(self.crossover(A=parent, B=next(parents_iterator), random_state=self.random_state_))

            children = [self.constraint(self.mutation(child, random_state=self.random_state_))
                        .fit(X, y) for child in children]

            # fill population for new iteration with 6/7 best children and 1/7 elitists
            population = []

            if self.first_iter:
                population.extend([x[0] for x in sorted(self._calculate_novelty_score(rules=children, archive=children, k=15),
                                                        key=lambda x: x[1], reverse=True)][
                                  :int(round(self.mu * 6 / 7))])
                population.extend([x[0] for x in sorted(self._calculate_novelty_score(rules=parents, archive=parents, k=15),
                                                        key=lambda x: x[1], reverse=True)][
                                  :int(round(self.mu * 1 / 7))])
            else:
                population.extend([x[0] for x in sorted(self._calculate_novelty_score(rules=children, archive=self.pool_, k=15),
                                                        key=lambda x: x[1], reverse=True)][:int(round(self.mu * 6 / 7))])
                population.extend([x[0] for x in sorted(self._calculate_novelty_score(rules=parents, archive=self.pool_, k=15),
                                                        key=lambda x: x[1], reverse=True)][:int(round(self.mu * 1 / 7))])

            if self.first_iter and i == self.iterations-1:
                self.first_iter = False

        return population

    def _calculate_novelty_score(self, rules: list[Rule], archive: list[Rule], k: int) -> list[Rule, float]:
        rules_with_novelty_score = []
        for rule in rules:
            distances = [jaccard(rule.match_, B.match_) for B in archive]
            distances.sort()
            novelty_score = sum(distances[:15]) / 15
            rules_with_novelty_score.append((rule, novelty_score))

        return rules_with_novelty_score

