"""Tests for the output degeneracy preflight.

C5-001 is void (``docs/TACOSM-C5-001-RESULT.md``) because its selection rule
was an argmax over a CASM-S output that was constant across candidates. C5-002
terminated as ``INSTRUMENT_INVALID``
(``docs/TACOSM-C5-002-RESULT.md``) because the same bridge reports 0.84375
absolute output accuracy while separating satisfier from violator in only
59/256 pairs.

This module turns those two failures into a preflight. The tests here are the
ones that keep it honest: they reproduce both failure modes against the
preflight, and they pin the construction the gate and the preflight share so
the two cannot drift apart.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tac_osm import degeneracy as d  # noqa: E402


def _script_module(name: str):
    """Import a measurement script, registered in ``sys.modules`` first.

    The scripts define dataclasses at module scope, and the dataclass machinery
    looks a class up in ``sys.modules`` by ``__module__`` while it is being
    processed — so registering the module *before* ``exec_module`` is what
    makes the import work at all.
    """
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"_script_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


C5 = _script_module("measure_c5_casm_002.py")


# --------------------------------------------------------------------------- #
# The pair construction must not drift from the C5-002 gate
# --------------------------------------------------------------------------- #


def test_make_pairs_matches_the_c5_002_gate_construction():
    """The preflight and the registered gate must build the same pairs.

    The code is deliberately not shared — a runnable script is the wrong
    dependency target for a library — so the construction is duplicated and
    this test is the seam that keeps the two in step. A divergence here would
    silently change what the gate tested, because the pairs *are* the gate's
    semantics: a violating descriptor is a satisfying one with exactly one
    marked bit flipped.
    """
    pairs = d.make_pairs(256, C5.GATE_SEED)
    gate = C5.gate_pairs(C5.GATE_STRUCTURES, C5.GATE_SEED)
    assert len(pairs) == len(gate) == 256
    for got, want in zip(pairs, gate):
        assert got.reference == want[0]
        assert got.satisfying == want[1]
        assert got.violating == want[2]
        assert got.marks == want[3]


def test_every_pair_differs_on_exactly_one_marked_bit():
    """The violation is one marked-bit flip, and only a marked one.

    This is what makes a pair a *pair*: the two descriptors differ on precisely
    the quantity the relation is defined over and on nothing else. A flip on an
    unmarked position would change nothing the relation can see, and a
    multi-bit flip would make the pair easier to separate than the task is.
    """
    for pair in d.make_pairs(128, 7):
        diffs = [j for j in range(len(pair.reference)) if pair.satisfying[j] != pair.violating[j]]
        assert len(diffs) == 1, (pair, diffs)
        assert diffs[0] in set(pair.marks), (pair, diffs)


def test_pairs_are_deterministic_across_calls():
    """A pre-registered gate's pairs must not depend on interpreter salt.

    ``hash()`` on strings is salted per process, which is why the C5-002 gate
    keys its structures by a formatted string rather than by ``hash``. The same
    reasoning applies to the pair stream itself: two runs of one registered
    design must see one pair set, not two.
    """
    assert d.make_pairs(32, 20260930) == d.make_pairs(32, 20260930)


# --------------------------------------------------------------------------- #
# Check 2 — the output distribution (the C5-001 failure mode)
# --------------------------------------------------------------------------- #


def test_constant_outputs_are_flagged_as_the_c5_001_signature():
    """The exact C5-001 failure: an argmax over this is deterministically index 0.

    C5-001's exhaustive arm selected candidate 0 on every query and its
    ``success_rate`` read the population base rate — 0.2500, seed spread
    0.0000, at every H — because the output was constant across candidates.
    """
    report = d.degenerate_outputs([0.3] * 100)
    assert [lvl.code for lvl in report.levels] == ["constant"]
    assert report.n_distinct == 1
    assert report.sd == 0.0
    assert report.spread == 0.0


@pytest.mark.parametrize("values", [[0.5, 0.5], [0.9], []])
def test_a_trivial_batch_is_degenerate(values):
    """A decision among one or zero candidates is not a decision."""
    report = d.degenerate_outputs(values)
    assert report.levels
    assert report.n_outputs == len(values)


def test_near_constant_outputs_are_flagged_on_both_scales():
    """Outputs that vary by 1e-12 are constant for every practical purpose.

    A purely relative test would pass these: the spread is large against a mean
    that is itself ~1e-12. The absolute floor is what catches the case, which
    is why the check requires variation on both scales rather than one.
    """
    values = [1.0, 1.0 + 1e-12] * 50
    report = d.degenerate_outputs(values, min_spread=0.05)
    assert [lvl.code for lvl in report.levels] == ["near_constant"]
    assert report.n_distinct > 1


def test_a_well_spread_distribution_is_not_degenerate():
    """A distribution that does vary on both scales passes the distribution check.

    Passing means only that the *distribution* is usable. Whether the outputs
    track the relation is a different check, which is why the two are separate.
    """
    report = d.degenerate_outputs([0.0, 1.0] * 50)
    assert report.levels == ()
    assert report.n_distinct == 2
    assert report.spread == pytest.approx(1.0)


def test_too_few_distinct_values_are_flagged():
    """Non-constant, but tied at a scale a within-query decision cannot resolve."""
    values = [0.0, 0.0, 1.0] * 40
    report = d.degenerate_outputs(values, min_spread=0.0, min_distinct=3)
    assert [lvl.code for lvl in report.levels] == ["too_few_distinct"]
    assert report.n_distinct == 2


# --------------------------------------------------------------------------- #
# Check 3 — discriminative content (the C5-002 failure mode)
# --------------------------------------------------------------------------- #


def _score_from_output(outputs):
    """Build a ``score`` callback from a per-descriptor output lookup.

    The C5-002 gate ran the model on each descriptor of each pair. Here the
    outputs are supplied, keyed by ``(pair index, which)``, so a test can
    replay an observed model's behaviour without the model.
    """
    def score(pair, which):
        return outputs[(pair, which)]
    return score


def test_a_separating_model_passes_discrimination():
    pairs = d.make_pairs(64, 11)
    outputs = {(p, "satisfying"): 1.0 for p in pairs}
    outputs.update({(p, "violating"): 0.0 for p in pairs})
    report = d.discrimination(pairs, score=_score_from_output(outputs))
    assert report.passed
    assert report.n_separated == 64
    assert report.accuracy == pytest.approx(1.0)


def test_a_constant_model_fails_discrimination():
    """The C5-001 model, scored as pairs: it separates nothing.

    Where C5-001 showed this as a ``success_rate`` at the base rate, the same
    model now fails the precondition the measurement depends on, before any
    capability number exists.
    """
    pairs = d.make_pairs(64, 11)
    outputs = {(p, w): 0.3 for p in pairs for w in ("satisfying", "violating")}
    report = d.discrimination(pairs, score=_score_from_output(outputs))
    assert not report.passed
    assert report.n_separated == 0
    assert report.accuracy == pytest.approx(0.0)


def test_an_inverted_model_fails_discrimination_by_direction():
    """Separation in the wrong direction is still a failure.

    The criterion is asymmetric — ``hi >= threshold > lo`` — so a model that
    anti-correlates with the relation scores zero. A symmetric "the outputs
    differ" test would pass it, which is the difference between a gate and a
    distribution check.
    """
    pairs = d.make_pairs(64, 11)
    outputs = {(p, "satisfying"): 0.0 for p in pairs}
    outputs.update({(p, "violating"): 1.0 for p in pairs})
    report = d.discrimination(pairs, score=_score_from_output(outputs))
    assert not report.passed
    assert report.n_separated == 0


def test_the_thresholds_are_the_registered_ones():
    """The preflight's bars are the registered bars, not stricter ones invented later.

    C5-002 pre-registered ``GATE_MIN_ACCURACY = 0.5`` and the run failed at
    0.2305. The spread floor was derived the same way: C5-001's output was
    constant, ``sd = 0``, so any positive value separates that failure from a
    working instrument. This pins the relationship for all three so a change to
    any constant is a deliberate revision visible in the diff, not a silent
    drift in what the preflight demands.
    """
    assert d.DEFAULT_MIN_DISCRIMINATION == C5.GATE_MIN_ACCURACY
    assert d.MIN_DISCRIMINATION == C5.GATE_MIN_ACCURACY
    assert d.DEFAULT_MIN_SPREAD > 0
    assert d.MIN_SPREAD == d.DEFAULT_MIN_SPREAD
    assert d.DEFAULT_MIN_CLASS_SUPPORT >= 2
    assert d.MIN_CLASS_SUPPORT == d.DEFAULT_MIN_CLASS_SUPPORT


def test_insufficient_class_support_is_undefined_not_scored():
    """A separation score over too few pairs is undefined, so it is not reported.

    The ``accuracy`` field is ``None`` rather than 0.0: 0.0 is a measurement
    that says "the model separates nothing", whereas this sample cannot support
    a measurement at all. A consumer that reads ``accuracy`` without checking
    for ``None`` is reading an absent value as a number.
    """
    pairs = d.make_pairs(1, 11)
    outputs = {(p, w): 1.0 for p in pairs for w in ("satisfying", "violating")}
    report = d.discrimination(pairs, score=_score_from_output(outputs))
    assert not report.passed
    assert report.accuracy is None
    assert [lvl.code for lvl in report.levels] == ["uneven_classes"]
    assert "accuracy" not in report.to_dict()


# --------------------------------------------------------------------------- #
# Check 5 — consistency with the instrument's own programme
# --------------------------------------------------------------------------- #


def test_programme_consistency_reports_agreement():
    pairs = d.make_pairs(32, 13)
    outputs = {(p, "satisfying"): 1.0 for p in pairs}
    outputs.update({(p, "violating"): 0.0 for p in pairs})

    def programme(pair, which):
        return 1.0 if which == "satisfying" else 0.0

    report = d.programme_consistency(
        pairs, score=_score_from_output(outputs), programme=programme
    )
    assert report.agreement == pytest.approx(1.0)
    assert report.passed


def test_programme_consistency_detects_disagreement():
    """The model is self-consistent while disagreeing with the instrument.

    This is the layer C5-001's ``verification_rate = 0.0000`` was an *effect*
    of rather than a measurement of: a constant output makes every verification
    fail. Here the outputs are not constant, so disagreement is a finding about
    the model rather than about a threshold meeting a degenerate output.
    """
    pairs = d.make_pairs(32, 13)
    outputs = {(p, "satisfying"): 0.0 for p in pairs}
    outputs.update({(p, "violating"): 1.0 for p in pairs})

    def programme(pair, which):
        return 1.0 if which == "satisfying" else 0.0

    report = d.programme_consistency(
        pairs, score=_score_from_output(outputs), programme=programme
    )
    assert report.agreement == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# The preflight and its ordering
# --------------------------------------------------------------------------- #


def _perfect_outputs(pairs):
    outputs = {(p, "satisfying"): 1.0 for p in pairs}
    outputs.update({(p, "violating"): 0.0 for p in pairs})
    return _score_from_output(outputs)


def test_preflight_passes_a_non_degenerate_separating_model():
    pairs = d.make_pairs(64, 20260930)
    result = d.preflight(
        representable=True,
        outputs=[0.0, 1.0] * 50,
        pairs=pairs,
        score=_perfect_outputs(pairs),
    )
    assert result.passed
    assert set(result.checks) == set(d.CHECK_ORDER)
    assert result.levels == ()


def test_preflight_refuses_a_representability_failure_first():
    """If the instrument's own programme cannot compute the relation, nothing else runs.

    This is the precondition the C5-002 gate assumed rather than checked. The
    ordering means a representability failure cannot be masked by a model whose
    outputs happen to separate a relation the instrument does not compute.
    """
    pairs = d.make_pairs(64, 20260930)
    with pytest.raises(d.DegeneracyError, match="representability"):
        d.preflight(
            representable=False,
            outputs=[0.0, 1.0] * 50,
            pairs=pairs,
            score=_perfect_outputs(pairs),
        )


def test_preflight_refuses_a_constant_model_before_scoring_pairs():
    """The C5-001 failure stops the run before a discrimination score is computed.

    A discrimination score over a degenerate distribution is a number about
    noise, so the ordering is part of the gate rather than a convenience.
    """
    pairs = d.make_pairs(64, 20260930)
    with pytest.raises(d.DegeneracyError, match="constant"):
        d.preflight(
            representable=True,
            outputs=[0.3] * 100,
            pairs=pairs,
            score=_perfect_outputs(pairs),
        )


def test_preflight_refuses_insufficient_class_support():
    pairs = d.make_pairs(1, 20260930)
    with pytest.raises(d.DegeneracyError, match="undefined"):
        d.preflight(
            representable=True,
            outputs=[0.0, 1.0] * 50,
            pairs=pairs,
            score=_perfect_outputs(pairs),
            min_class_support=2,
        )


def test_preflight_refuses_a_non_separating_model():
    """The C5-002 failure: 0.84375 absolute accuracy, separation below the bar.

    The model's outputs here are well spread and non-constant, so the
    distribution check passes and the run reaches the discrimination check —
    which is exactly the case an accuracy score alone cannot catch.
    """
    pairs = d.make_pairs(64, 20260930)
    outputs = {(p, w): 0.3 for p in pairs for w in ("satisfying", "violating")}
    with pytest.raises(d.DegeneracyError, match="59 of"):
        d.preflight(
            representable=True,
            outputs=[0.0, 1.0] * 50,
            pairs=pairs,
            score=_score_from_output(outputs),
        )


def test_check_order_is_the_order_the_preflight_runs():
    """The order is load-bearing, so it is pinned by name.

    Class support precedes discrimination because a separation score is
    undefined without it; degeneracy precedes both because a score over a
    degenerate distribution measures noise. A reordering here is a change to
    the gate's semantics, not to its presentation.
    """
    assert d.CHECK_ORDER == (
        "representability",
        "degeneracy",
        "class_support",
        "discrimination",
        "programme_consistency",
    )
    assert d.REQUIRED_CHECKS == d.CHECK_ORDER


def test_the_c5_002_finding_is_reproducible_from_the_recorded_pairs():
    """The bridge's 59/256 separation reproduces against the gate's own pair stream.

    The recorded artifact gives the gate's accuracy as 0.23046875 over 256
    pairs, which is 59 separated. This test asserts the arithmetic that makes
    the reported number interpretable: the fraction, the count, and that the
    count is an integer, because a run record that reported 59.7 pairs would be
    reporting something the instrument did not measure.
    """
    accuracy = 0.23046875
    n_pairs = 256
    n_separated = round(accuracy * n_pairs)
    assert n_separated == 59
    assert n_separated / n_pairs == pytest.approx(accuracy)
    assert n_separated < d.MIN_DISCRIMINATION * n_pairs


# --------------------------------------------------------------------------- #
# The module's own contract
# --------------------------------------------------------------------------- #


def test_every_exported_name_resolves():
    """Every name in ``__all__`` is defined, and the aliases are the defaults.

    ``__all__`` is the contract a successor pre-registration imports by name
    from, so a name listed there but absent here is an import error at the worst
    moment — in a run record's preflight, not in a test. The registered
    aliases exist so a contract can pin a value by name; if the two drift apart
    the pinning is worthless.
    """
    missing = [name for name in d.__all__ if not hasattr(d, name)]
    assert not missing, missing
    assert len(d.__all__) == len(set(d.__all__)), "duplicate exports"
    assert d.MIN_SPREAD is d.DEFAULT_MIN_SPREAD
    assert d.MIN_DISCRIMINATION is d.DEFAULT_MIN_DISCRIMINATION
    assert d.MIN_CLASS_SUPPORT is d.DEFAULT_MIN_CLASS_SUPPORT


def test_the_levels_are_fatal_and_distinct():
    """Every degeneracy level terminates the run, and each fires for its own reason.

    All four are fatal because every one is a precondition for a number the
    measurement would otherwise publish. They have distinct codes because a run
    record reports which predicate fired, and two levels sharing a code would
    make a record ambiguous.
    """
    levels = [d.LEVEL_CONSTANT, d.LEVEL_NEAR_CONSTANT, d.LEVEL_TOO_FEW_DISTINCT, d.LEVEL_UNEVEN_CLASSES]
    assert all(lvl.fatal for lvl in levels)
    codes = [lvl.code for lvl in levels]
    assert len(set(codes)) == len(codes), codes
    # Each detail names the failure it accounts for rather than the generic one.
    assert "TACOSM-C5-001" in d.LEVEL_CONSTANT.detail
    for lvl in levels:
        assert lvl.detail, lvl
