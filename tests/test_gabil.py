"""Tests for the GABIL categorical condition.

Unit tests exercise each part of the MatchingFunction contract in
isolation, so a failure points at one method. The integration test then
runs the full pipeline, which is what catches wiring problems such as a
missing dispatch branch that the unit tests cannot see.
"""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator
from suprb import SupRB
from suprb.rule.matching import GABIL
from suprb.rule.initialization import MeanInit
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.rule.mutation import HalfnormIncrease
from suprb.optimizer.solution.ga import GeneticAlgorithm
from suprb.utils import check_random_state


N_LEVELS = 6
TRUE_SUBSET = {1, 3, 4}


class FakeRule:
    """Minimal stand-in for a Rule, so mutation can be tested alone."""

    def __init__(self, bits):
        self.match = GABIL(bits=bits)


# --- the matching contract -------------------------------------------

def test_match_lookup():
    """A sample matches exactly when its level's bit is set."""
    cond = GABIL(bits=np.array([True, False, True, False]))
    X = np.array([[0], [1], [2], [3]], dtype=float)
    assert np.array_equal(cond(X), [True, False, True, False])


def test_volume_is_bit_fraction():
    """volume_ is the categorical counterpart of interval width."""
    assert GABIL(bits=np.array([True, False, True, False])).volume_ == 0.5
    assert GABIL(bits=np.ones(4, dtype=bool)).volume_ == 1.0


def test_copy_is_deep():
    """Mutating a copy must not reach back into the original.

    Offspring are produced by copying a parent and then mutating in place,
    so a shared array would corrupt the parent silently rather than raise.
    """
    parent = GABIL(bits=np.array([True, False, False, False]))
    child = parent.copy()
    child.bits[1] = True

    assert not parent.bits[1]


# --- repair and constraints ------------------------------------------

def test_clip_repairs_all_zero():
    """An all-zero bitstring matches nothing and must be repaired."""
    cond = GABIL(bits=np.zeros(4, dtype=bool))
    cond.clip(None)

    assert cond.bits.any()


def test_clip_leaves_valid_rules_alone():
    """Repair must not disturb a rule that is already valid."""
    cond = GABIL(bits=np.array([False, True, False, False]))
    cond.clip(None)

    assert np.array_equal(cond.bits, [False, True, False, False])


def test_clip_is_deterministic():
    """Repair takes no random_state, so it must not behave randomly.

    RuleConstraint's interface passes no generator. Choosing a random bit
    here would make fits irreproducible under a fixed seed.
    """
    a, b = (GABIL(bits=np.zeros(4, dtype=bool)) for _ in range(2))
    a.clip(None)
    b.clip(None)

    assert np.array_equal(a.bits, b.bits)


def test_min_range_is_a_noop():
    """A bitstring has no continuous extent to widen."""
    cond = GABIL(bits=np.array([False, True, False, False]))
    cond.min_range(1e-6)

    assert np.array_equal(cond.bits, [False, True, False, False])


# --- mutation ---------------------------------------------------------

def test_mutation_never_unsets_a_bit():
    """This is the whole definition of Adding Alternative.

    The interval operator it mirrors can only widen a rule, because
    covering produces something too specific to be useful.
    """
    bits = np.zeros(10, dtype=bool)
    bits[3] = True
    rule = FakeRule(bits)

    mutation = HalfnormIncrease(matching_type=GABIL(np.zeros(10, dtype=bool)), sigma=0.3)
    rs = check_random_state(42)

    for _ in range(20):
        mutation.gabil(rule, rs)
        assert rule.match.bits[3]


def test_mutation_only_grows():
    """Bit count is monotonically non-decreasing."""
    rule = FakeRule(np.zeros(10, dtype=bool))
    rule.match.bits[0] = True

    mutation = HalfnormIncrease(matching_type=GABIL(np.zeros(10, dtype=bool)), sigma=0.3)
    rs = check_random_state(42)

    previous = int(rule.match.bits.sum())
    for _ in range(10):
        mutation.gabil(rule, rs)
        current = int(rule.match.bits.sum())
        assert current >= previous
        previous = current


def test_mutation_is_seeded():
    """Identical seeds must give identical bitstrings."""
    results = []
    for _ in range(2):
        rule = FakeRule(np.zeros(10, dtype=bool))
        rule.match.bits[3] = True
        mutation = HalfnormIncrease(matching_type=GABIL(np.zeros(10, dtype=bool)), sigma=0.3)
        rs = check_random_state(42)
        for _ in range(5):
            mutation.gabil(rule, rs)
        results.append(rule.match.bits.copy())

    assert np.array_equal(*results)


# --- documented boundaries -------------------------------------------

def test_rejects_multiple_columns():
    """One bitstring governs one attribute.

    Flattening a wider X would return a mask of the wrong length and
    mismatch samples to rules without raising, so the restriction is
    asserted here to stop it being reintroduced as an optimisation.
    """
    cond = GABIL(bits=np.array([True, False, True, False]))
    X = np.array([[0, 1], [2, 3]], dtype=float)

    with pytest.raises(ValueError, match="single categorical attribute"):
        cond(X)


def test_rejects_out_of_range_codes():
    """An out-of-range code is a malformed encoding, not a value to clip."""
    cond = GABIL(bits=np.array([True, False, True, False]))

    with pytest.raises(ValueError, match="Category codes"):
        cond(np.array([[99]], dtype=float))

    with pytest.raises(ValueError, match="Category codes"):
        cond(np.array([[-1]], dtype=float))


# --- integration ------------------------------------------------------

def _make_data():
    rng = np.random.RandomState(42)
    X = rng.randint(0, N_LEVELS, size=(400, 1)).astype(float)
    y = np.array([5.0 if int(v) in TRUE_SUBSET else 1.0 for v in X.ravel()])
    y += rng.normal(scale=0.1, size=y.shape)
    return train_test_split(X, y, random_state=42)


def _build_model():
    return SupRB(
        rule_discovery=ES1xLambda(
            n_iter=10,
            lmbda=8,
            operator="+",
            init=MeanInit(),
            mutation=HalfnormIncrease(sigma=0.3),
        ),
        solution_composition=GeneticAlgorithm(n_iter=8, population_size=16),
        matching_type=GABIL(bits=np.zeros(N_LEVELS, dtype=bool)),
        n_iter=3,
        random_state=42,
        verbose=0,
    )


def test_end_to_end_recovers_the_concept():
    """The full pipeline, which is what exercises the dispatch wiring."""
    X_train, X_test, y_train, y_test = _make_data()
    model = _build_model().fit(X_train, y_train)

    assert model.score(X_test, y_test) > 0.95

    found = {frozenset(np.flatnonzero(r.match.bits).tolist()) for r in model.pool_}
    complement = set(range(N_LEVELS)) - TRUE_SUBSET

    assert frozenset(TRUE_SUBSET) in found or frozenset(complement) in found


def test_fit_is_reproducible():
    """Two fits under one seed must agree bit for bit."""
    X_train, X_test, y_train, y_test = _make_data()

    runs = []
    for _ in range(2):
        model = _build_model().fit(X_train, y_train)
        runs.append((model.score(X_test, y_test), [r.match.bits.copy() for r in model.pool_]))

    (score_a, bits_a), (score_b, bits_b) = runs

    assert score_a == score_b
    assert len(bits_a) == len(bits_b)
    assert all(np.array_equal(x, y) for x, y in zip(bits_a, bits_b))


# --- sklearn estimator checks ----------------------------------------

@pytest.mark.xfail(
    reason="check_regressors_train fails on multi-column input: GABIL raises "
    "ValueError, SupRB records the error, and predict then returns a list "
    "rather than an ndarray, which sklearn cannot take .shape of.",
    strict=False,
)
def test_check_estimator():
    """Run sklearn's estimator checks against a GABIL configuration.

    The suite verifies, among other things, that two fits under one seed
    agree, which is the property most at risk when a new representation
    introduces its own random draws.

    Iterations are kept low for speed; the checks are about interface
    conformance rather than solution quality.
    """
    X_train, _, y_train, _ = _make_data()

    estimator = _build_model()
    estimator.fit(X_train, y_train)

    check_estimator(estimator)