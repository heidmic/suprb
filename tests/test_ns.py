from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle as apply_shuffle

from suprb import SupRB
from suprb import rule
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import ns
from suprb.rule.matching import OrderedBound
from suprb.utils import check_random_state
import unittest

import numpy as np

from suprb.utils import check_random_state
from suprb.optimizer.rule.ns.novelty_calculation import (
    NoveltyCalculation,
    ProgressiveMinimalCriteria,
    NoveltyFitnessBiased,
    NoveltyFitnessPareto,
)
from suprb.optimizer.rule.ns.novelty_search_type import (
    NoveltySearchType,
    LocalCompetition,
    MinimalCriteria,
)
from suprb.optimizer.rule.ns.archive import ArchiveNone, ArchiveNovel, ArchiveRandom
import inspect
import itertools


class TestNoveltySearch(unittest.TestCase):
    """Test Novelty Search Implementation based on Higdon Gramacy Lee"""

    def setup_test_example(self):

        n_samples = 1000
        random_state = 42
        random_state_ = check_random_state(random_state)

        X = np.linspace(0, 20, num=n_samples)
        y = np.zeros(n_samples)
        y[X < 10] = np.sin(np.pi * X[X < 10] / 5) + 0.2 * np.cos(
            4 * np.pi * X[X < 10] / 5
        )
        y[X >= 10] = X[X >= 10] / 10 - 1
        y += random_state_.normal(scale=0.1, size=n_samples)
        X = X.reshape((-1, 1))
        X, y = apply_shuffle(X, y, random_state=random_state)

        self.X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
        self.y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    def setup_base_model(self):
        from suprb import optimizer

        self.model = SupRB(
            rule_discovery=ns.NoveltySearch(
                init=rule.initialization.HalfnormInit(),
                selection=optimizer.rule.selection.RouletteWheel(),
            ),
            solution_composition=ga.GeneticAlgorithm(),
            matching_type=OrderedBound(np.array([])),
            n_iter=2,
            n_rules=8,
            verbose=10,
            logger=CombinedLogger([("stdout", StdoutLogger()), ("default", DefaultLogger())]),
        )

    def setup_novelty_search_params(self):

        novelty_calculation_types = [
            NoveltyCalculation,
            ProgressiveMinimalCriteria,
            NoveltyFitnessPareto,
            NoveltyFitnessBiased,
        ]

        novelty_search_types = [
            NoveltySearchType(),
            MinimalCriteria(min_examples_matched=10),
            LocalCompetition(max_neighborhood_range=15),
        ]

        archive_types = [ArchiveNovel(), ArchiveRandom(), ArchiveNone()]

        self.combined_ns_params = list(
            itertools.product(novelty_calculation_types, novelty_search_types, archive_types)
        )

    def setUp(self) -> None:
        self.setup_test_example()
        self.setup_novelty_search_params()

        return super().setUp()

    def filter_kwargs(self, dict_to_filter, kwargs_dict):
        sig = inspect.signature(kwargs_dict)
        filter_keys = [
            param.name
            for param in sig.parameters.values()
            if param.kind == param.POSITIONAL_OR_KEYWORD
        ]
        return {filter_key: dict_to_filter[filter_key] for filter_key in filter_keys}

    def setup_kwargs(self, ns_search_type, archive_type, novelty_calculation_type):
        kwargs_dict = dict(
            novelty_bias=0.33,
            novelty_search_type=ns_search_type,
            archive=archive_type,
            k_neighbor=20,
        )

        self.kwargs = self.filter_kwargs(kwargs_dict, novelty_calculation_type)

    def test_smoke_test(self):
        for (
            novelty_calculation_type,
            ns_search_type,
            archive_type,
        ) in self.combined_ns_params:
            self.setup_base_model()
            self.setup_kwargs(ns_search_type, archive_type, novelty_calculation_type)

            print(f"\n\nChecking... {novelty_calculation_type.__name__} {self.kwargs}")

            try:
                self.model.rule_discovery.novelty_calculation = (
                    novelty_calculation_type(**self.kwargs)
                )
                self.assertTrue(True),
                print("PASSED [Model Generation]\n")
            except:
                self.assertTrue(
                    False
                ), f"FAILED! Model generation with this config: {self.kwargs}"

            try:
                self.model.fit(self.X, self.y)
                self.assertTrue(True),
                print("PASSED [Model fit]\n")
            except:
                self.assertTrue(
                    False
                ), f"FAILED! Model fit with this config: {self.kwargs}"


if __name__ == "__main__":
    unittest.main()
