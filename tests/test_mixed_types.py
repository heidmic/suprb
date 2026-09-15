"""Tests for the MixedTypes composed condition and the encoding paths.

MixedTypes routes each attribute to whichever representation suits it, so
most of what is tested here is the routing rather than the matching: that
each half receives only its own columns, that both halves survive a copy
independently, and that what a bit stands for is carried through
initialisation, crossover and serialisation.

The encoding tests cover the three ways a level count can reach fit, since
they differ in what they can know about levels absent from the training
split rather than in what they compute.
"""

import warnings

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from suprb import SupRB
from suprb.json import _convert_match_to_json, _convert_matching_type
from suprb.optimizer.rule.crossover import UniformCrossover
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.rule.mutation import HalfnormIncrease
from suprb.optimizer.solution.ga import GeneticAlgorithm
from suprb.rule import Rule
from suprb.rule.fitness import VolumeWu
from suprb.rule.initialization import MeanInit
from suprb.rule.matching import GABIL, MixedTypes, OrderedBound
from suprb.utils import check_random_state

CITIES = np.array(["Aachen", "Berlin", "Karachi", "Lahore"])


def _mixed(real_bounds, bits, mask=(False, True), observed=None):
    """A MixedTypes with one real and one categorical attribute by default."""
    return MixedTypes(
        reals=OrderedBound(np.array(real_bounds, dtype=float)),
        categoricals=GABIL(bits=[np.array(b, dtype=bool) for b in bits], observed=observed),
        mask=np.array(mask, dtype=bool),
    )


def _rule(match, n_features=2):
    return Rule(
        match=match,
        input_space=np.array([[-1.0, 1.0]] * n_features),
        model=Ridge(),
        fitness=VolumeWu(),
    )


# --- routing ----------------------------------------------------------


def test_each_half_sees_only_its_own_columns():
    """The interval half must not be handed a category code, or vice versa.

    A mask applied the wrong way round would still produce a boolean array
    of the right length, so this is asserted rather than left to the
    integration test to notice.
    """
    cond = _mixed(real_bounds=[[0.0, 0.5]], bits=[[True, False, True]])

    X = np.array(
        [
            [0.2, 0.0],  # real inside, category set
            [0.2, 1.0],  # real inside, category unset
            [0.9, 0.0],  # real outside, category set
            [0.9, 1.0],  # both fail
        ]
    )

    assert np.array_equal(cond(X), [True, False, False, False])


def test_mask_order_is_respected():
    """The categorical column is not assumed to come last."""
    cond = _mixed(real_bounds=[[0.0, 0.5]], bits=[[True, False]], mask=(True, False))

    X = np.array([[0.0, 0.2], [1.0, 0.2], [0.0, 0.9]])

    assert np.array_equal(cond(X), [True, False, False])


def test_rejects_wrong_number_of_columns():
    """A column-count mismatch means the input is not the trained data."""
    cond = _mixed(real_bounds=[[0.0, 1.0]], bits=[[True, False]])

    with pytest.raises(ValueError, match="built for 2 attributes"):
        cond(np.array([[0.1, 0.0, 0.0]]))


def test_accepts_a_mask_given_as_a_list():
    """Nothing in the contract requires the mask to be an ndarray."""
    cond = MixedTypes(
        reals=OrderedBound(np.array([[0.0, 1.0]])),
        categoricals=GABIL(bits=[np.array([True, False])]),
        mask=[False, True],
    )

    assert np.array_equal(cond(np.array([[0.5, 0.0], [0.5, 1.0]])), [True, False])


# --- the matching contract -------------------------------------------


def test_volume_is_the_product_of_both_halves():
    """Together the halves describe one region, so their volumes multiply."""
    cond = _mixed(real_bounds=[[0.0, 0.5]], bits=[[True, False, True, False]])

    assert cond.volume_ == pytest.approx(0.5 * 0.5)


def test_copy_is_deep_in_both_halves():
    """Mutation modifies a rule in place, so a child must not reach back.

    Both halves are checked, since sharing either one is enough to let a
    mutated child corrupt its parent, and a partly independent copy is the
    harder case to notice.
    """
    parent = _mixed(real_bounds=[[0.0, 0.5]], bits=[[True, False, False]])
    child = parent.copy()

    child.reals.bounds[0, 1] = 0.9
    child.categoricals.bits[0][2] = True

    assert parent.reals.bounds[0, 1] == 0.5
    assert not parent.categoricals.bits[0][2]


def test_clip_slices_bounds_down_to_the_real_columns():
    """Clip receives bounds for every attribute, including categorical ones.

    Passing the full array through would clip each interval against
    whichever feature happened to sit at that index.
    """
    cond = _mixed(real_bounds=[[-5.0, 5.0]], bits=[[True, False]])

    # Column 0 is real and bounded to [-1, 1]; column 1 is the categorical.
    cond.clip(np.array([[-1.0, 1.0], [0.0, 1.0]]))

    assert np.array_equal(cond.reals.bounds, [[-1.0, 1.0]])


def test_min_range_repairs_both_halves():
    """Matching ANDs the halves, so either one can make a rule match nothing."""
    cond = _mixed(real_bounds=[[0.5, 0.5]], bits=[[False, False, False]])
    cond.min_range(0.1)

    assert cond.reals.bounds[0, 1] > cond.reals.bounds[0, 0]
    assert cond.categoricals.bits[0].any()


# --- what each bit stands for ----------------------------------------


def test_observed_marks_levels_absent_from_training():
    """A level the encoder knows but the split lacks gets a bit and no data.

    Recording it as None is what later lets a rule tell a level it was
    trained on from one it has no evidence for.
    """
    encoder = OrdinalEncoder().fit(CITIES.reshape(-1, 1))

    # Training contains every city except Aachen, which sorts first.
    rng = np.random.RandomState(42)
    codes = rng.randint(1, 4, size=200)
    X = encoder.transform(CITIES[codes].reshape(-1, 1))
    y = np.where(codes == 2, 5.0, 1.0)

    model = _build_model(GABIL(), categorical_encoder=encoder)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    observed = model.pool_[0].match.observed[0]

    assert observed[0] is None
    assert observed[1:] == ["Berlin", "Karachi", "Lahore"]


def test_absent_levels_are_reported_before_training():
    """Reported at fit rather than only at predict, and by name.

    A level missing from the split is usually a resplit away from being
    fixed, which the user can only act on if they know which one it is.
    """
    encoder = OrdinalEncoder().fit(CITIES.reshape(-1, 1))

    rng = np.random.RandomState(42)
    codes = rng.randint(1, 4, size=200)
    X = encoder.transform(CITIES[codes].reshape(-1, 1))
    y = np.where(codes == 2, 5.0, 1.0)

    with pytest.warns(UserWarning, match="Aachen"):
        _build_model(GABIL(), categorical_encoder=encoder).fit(X, y)


def test_unseen_level_warns_at_predict():
    """The rule matched a level no rule has evidence for."""
    cond = GABIL(
        bits=[np.array([True, True, True, True])],
        observed=[[None, "Berlin", "Karachi", "Lahore"]],
    )

    with pytest.warns(UserWarning, match="never appeared during fit"):
        cond(np.array([[0.0]]))


def test_known_levels_do_not_warn():
    """The warning must not fire for levels the training data contained."""
    cond = GABIL(
        bits=[np.array([True, True, True, True])],
        observed=[[None, "Berlin", "Karachi", "Lahore"]],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cond(np.array([[1.0], [2.0], [3.0]]))


def test_fallback_marks_gaps_in_the_codes():
    """max + 1 cannot see a missing level, but it can see a missing code.

    Codes 0, 1 and 3 allocate four bits, and bit 2 has no training data
    behind it even though it lies inside the range.
    """
    X = np.array([[0.0], [1.0], [3.0], [0.0], [1.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])

    model = _build_model(GABIL())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    observed = model.pool_[0].match.observed[0]

    assert len(observed) == 4
    assert observed[2] is None
    assert observed[0] == 0 and observed[3] == 3


# --- covering ---------------------------------------------------------


def test_covering_splits_the_sample_across_both_halves():
    """Each half is covered from the columns it governs, not from all of them."""
    init = MeanInit(
        bounds=np.array([[-1.0, 1.0], [0.0, 2.0]]),
        matching_type=_mixed(real_bounds=[[]], bits=[[False, False, False]]),
    )

    rule = init(check_random_state(42), mean=np.array([0.3, 2.0]))

    assert np.array_equal(rule.match.reals.bounds, [[0.3, 0.3]])
    assert np.array_equal(rule.match.categoricals.bits[0], [False, False, True])


def test_covering_rejects_a_fractional_category():
    """UniformInputOrigin produces a value no category corresponds to.

    Truncating it would pick a neighbour, and neighbouring means nothing
    for an attribute with no order.
    """
    init = MeanInit(
        bounds=np.array([[0.0, 3.0]]),
        matching_type=GABIL(bits=[np.zeros(4, dtype=bool)]),
    )

    with pytest.raises(TypeError, match="fractional value"):
        init(check_random_state(42), mean=np.array([2.37]))


# --- mutation ---------------------------------------------------------


def test_mutation_widens_both_halves():
    """Both halves generalise, as the interval variant does per dimension."""
    rule = _rule(_mixed(real_bounds=[[0.2, 0.2]], bits=[[True, False, False, False, False, False]]))

    mutation = HalfnormIncrease(matching_type=rule.match, sigma=3.0)
    rs = check_random_state(42)

    for _ in range(20):
        mutation.mixed_types(rule, rs)

    assert rule.match.reals.bounds[0, 1] > 0.2
    assert rule.match.categoricals.bits[0].sum() > 1


# --- crossover --------------------------------------------------------


def test_crossover_swaps_whole_bitstrings():
    """An attribute is inherited from one parent or the other, never mixed.

    Crossing inside a bitstring would produce a condition neither parent
    had, which is not what the interval variant does with a bound pair.
    """
    A = _rule(GABIL(bits=[np.array([True, False, False]), np.array([True, False])]), n_features=2)
    B = _rule(GABIL(bits=[np.array([False, True, True]), np.array([False, True])]), n_features=2)

    children = UniformCrossover(crossover_rate=1.0).gabil(A, B, check_random_state(42))

    for child in children:
        for i, bitstring in enumerate(child.match.bits):
            assert np.array_equal(bitstring, A.match.bits[i]) or np.array_equal(bitstring, B.match.bits[i])


def test_crossover_leaves_the_parents_alone():
    """Children are produced by copying, so a parent must not be edited."""
    A = _rule(_mixed(real_bounds=[[-1.0, 0.0]], bits=[[True, False, False]]))
    B = _rule(_mixed(real_bounds=[[0.5, 1.0]], bits=[[False, True, True]]))

    UniformCrossover(crossover_rate=1.0).mixed_types(A, B, check_random_state(42))

    assert np.array_equal(A.match.reals.bounds, [[-1.0, 0.0]])
    assert np.array_equal(A.match.categoricals.bits[0], [True, False, False])


def test_crossover_keeps_the_mask_and_observed():
    """Both describe the dataset, so crossover has no business changing them."""
    A = _rule(_mixed(real_bounds=[[-1.0, 0.0]], bits=[[True, False]], observed=[["x", "y"]]))
    B = _rule(_mixed(real_bounds=[[0.5, 1.0]], bits=[[False, True]], observed=[["x", "y"]]))

    children = UniformCrossover(crossover_rate=1.0).mixed_types(A, B, check_random_state(42))

    for child in children:
        assert np.array_equal(child.match.mask, [False, True])
        assert child.match.categoricals.observed == [["x", "y"]]


# --- serialisation ----------------------------------------------------


@pytest.mark.parametrize(
    "match, X",
    [
        (
            GABIL(
                bits=[np.array([True, False, True, False]), np.array([False, True])],
                observed=[[None, "Berlin", "Karachi", "Lahore"], ["large", "small"]],
            ),
            np.array([[0.0, 1.0], [2.0, 0.0]]),
        ),
        (
            _mixed(
                real_bounds=[[-1.0, 0.5]],
                bits=[[True, False, True, False]],
                observed=[[None, "Berlin", "Karachi", "Lahore"]],
            ),
            np.array([[0.0, 1.0], [-0.5, 2.0]]),
        ),
    ],
    ids=["GABIL", "MixedTypes"],
)
def test_serialisation_preserves_matching(match, X):
    """A restored rule must match exactly what it matched before.

    Bitstrings can differ in length and what each bit stands for may be a
    label, a code or None, so a round trip is the only way to be sure none
    of that is coerced away.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = match(X)

        restored = _convert_matching_type(
            _convert_match_to_json(match),
            f"class:suprb.rule.matching.{type(match).__name__}",
        )
        after = restored(X)

    assert np.array_equal(before, after)


def test_serialisation_keeps_none_in_observed():
    """None marks a level with no training data, and is not a string."""
    match = GABIL(bits=[np.array([True, False])], observed=[[None, "Berlin"]])

    restored = _convert_matching_type(_convert_match_to_json(match), "class:suprb.rule.matching.GABIL")

    assert restored.observed[0][0] is None


# --- integration ------------------------------------------------------


def _build_model(matching_type, **kwargs):
    return SupRB(
        rule_discovery=ES1xLambda(
            n_iter=10,
            lmbda=8,
            operator="+",
            init=MeanInit(),
            mutation=HalfnormIncrease(sigma=3.0),
        ),
        solution_composition=GeneticAlgorithm(n_iter=8, population_size=16),
        matching_type=matching_type,
        n_iter=3,
        random_state=42,
        verbose=0,
        **kwargs,
    )


def _make_mixed_data():
    """One real attribute and one categorical, both carrying signal."""
    rng = np.random.RandomState(42)

    real = rng.uniform(-1, 1, size=(400, 1))
    codes = rng.randint(0, 4, size=400)

    X = np.hstack([real, codes.reshape(-1, 1).astype(float)])
    y = 3.0 * real.ravel() + np.where(np.isin(codes, [1, 3]), 5.0, 0.0)
    y += rng.normal(scale=0.1, size=y.shape)

    return train_test_split(X, y, random_state=42)


def test_end_to_end_recovers_the_categorical_concept():
    """The full pipeline, which is what exercises the dispatch wiring."""
    X_train, X_test, y_train, y_test = _make_mixed_data()

    model = _build_model(
        MixedTypes(
            reals=OrderedBound(np.array([])),
            categoricals=GABIL(),
            mask=np.array([False, True]),
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    assert model.score(X_test, y_test) > 0.95

    found = {frozenset(np.flatnonzero(r.match.categoricals.bits[0]).tolist()) for r in model.pool_}

    assert frozenset({1, 3}) in found or frozenset({0, 2}) in found


def test_raw_labels_are_encoded_internally():
    """Handing SupRB labels rather than codes should not change the outcome.

    The encoder is fitted once and only applied afterwards, so a label
    keeps the same code at predict time.
    """
    rng = np.random.RandomState(42)

    real = rng.uniform(-1, 1, size=(400, 1))
    codes = rng.randint(0, 4, size=400)

    X = np.empty((400, 2), dtype=object)
    X[:, 0] = real.ravel()
    X[:, 1] = CITIES[codes]

    y = 3.0 * real.ravel() + np.where(np.isin(codes, [1, 3]), 5.0, 0.0)
    y += rng.normal(scale=0.1, size=y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = _build_model(
        MixedTypes(
            reals=OrderedBound(np.array([])),
            categoricals=GABIL(),
            mask=np.array([False, True]),
        ),
        encode_categoricals=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    assert model.score(X_test, y_test) > 0.95
    assert list(model.fitted_encoder_.categories_[0]) == list(CITIES)


def test_fit_is_reproducible():
    """Two fits under one seed must agree bit for bit."""
    X_train, _, y_train, _ = _make_mixed_data()

    runs = []
    for _ in range(2):
        model = _build_model(
            MixedTypes(
                reals=OrderedBound(np.array([])),
                categoricals=GABIL(),
                mask=np.array([False, True]),
            )
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        runs.append([b.copy() for r in model.pool_ for b in r.match.categoricals.bits])

    a, b = runs

    assert len(a) == len(b)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
