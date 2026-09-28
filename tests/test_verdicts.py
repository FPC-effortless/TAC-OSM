"""Tests for the materiality verdicts.

The verdict is the pre-registered decision rule applied to one delta. It was
duplicated in ``measure_surrogate.py`` and ``measure_learn.py`` before this
module existed, each copy with its own spelling of the 0.02 floor — which is
exactly the drift class the contract module exists to prevent, one level down:
the same delta would read as material in one experiment and as noise in the
other, with nothing in either output to show it.

What these tests pin:

1. **The three verdicts are the only three** — a fourth value anywhere is a
   verdict the rule never committed to.
2. **The 0.02 floor holds** — a spread below it is sampling error, not the
   experiment's noise floor, and a threshold built from it would let a 1.5%
   difference be called material.
3. **The threshold is symmetric** — harm and improvement are the same
   distance from zero, so the rule cannot be read as lenient in one direction.
4. **The delta sign convention is fixed** — ``observed - baseline``, and a
   positive delta is an improvement. Two scripts disagreeing about the sign
   would invert every comparison between them without any visible error.
5. **The spread is max-minus-min** — not a standard deviation, not a
   mean-normalised range, and empty input is defined rather than raising.

Layer 0: no verdict here is a scientific conclusion. The verdict answers "did
this row fire?"; whether the hypothesis is true is a reader's judgement against
the pre-registration's *consequences*, which live in the contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tac_osm.measurement import verdicts  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = REPO_ROOT / "contracts"


# --------------------------------------------------------------------------- #
# The verdicts are closed
# --------------------------------------------------------------------------- #


def test_the_three_verdicts_are_named_constants():
    """The rule's outputs are named, not ad-hoc strings per call site.

    A verdict spelled inline is a verdict a typo silently changes, and one
    spelled differently in two places is two different rules under one name.
    """
    assert verdicts.MATERIAL_IMPROVEMENT == "MATERIAL improvement"
    assert verdicts.MATERIAL_HARM == "MATERIAL harm"
    assert verdicts.WITHIN_SEED_NOISE == "within seed noise"


def test_the_registered_verdicts_are_the_only_ones():
    """A fourth verdict is one the rule never committed to.

    The set is what a reader checks a record against: a payload carrying
    anything else is a payload the pre-registration does not cover, which is
    a stronger statement than "unexpected value".
    """
    registered = {
        verdicts.MATERIAL_IMPROVEMENT,
        verdicts.MATERIAL_HARM,
        verdicts.WITHIN_SEED_NOISE,
    }
    # Every verdict the module can produce is reached from ``verdict()``,
    # which is the single entry point the scripts use.
    produced = {
        verdicts.verdict(d, s)
        for d in (-1.0, -0.5, -0.1, -0.02, -0.019, 0.0, 0.019, 0.02, 0.1, 0.5, 1.0)
        for s in (0.0, 0.01, 0.02, 0.1, 0.5)
    }
    assert produced == registered, (
        f"the verdict function produces values outside the registered set: "
        f"{produced - registered}"
    )


# --------------------------------------------------------------------------- #
# The 0.02 floor
# --------------------------------------------------------------------------- #


def test_the_floor_is_a_registered_constant():
    """The floor is written down once, in the module, not per script.

    ``0.02`` appearing as a bare literal in a measurement script is a
    pre-registered constant that can be edited without leaving a trace; the
    name is what keeps it visible as a *choice*.
    """
    assert verdicts.SPREAD_FLOOR == 0.02
    assert verdicts.spread_floor() == verdicts.SPREAD_FLOOR


@pytest.mark.parametrize("spread", [0.0, 0.005, 0.01, 0.015, 0.0199])
def test_a_sub_floor_spread_is_floored(spread):
    """A spread below the floor is sampling error, not a noise floor.

    The floor's reason is the evaluation length: at 100 evaluation steps a
    seed spread under 2% is the binomial sampling error of the estimate, not
    the experiment's reproducibility. Using it as the threshold would call a
    1.5% difference material, which is a claim about the seed draw rather than
    about the intervention.
    """
    assert verdicts.materiality_threshold(spread) == verdicts.SPREAD_FLOOR


@pytest.mark.parametrize("spread", [0.02, 0.0201, 0.05, 0.2, 1.0])
def test_a_real_spread_is_used_as_the_threshold(spread):
    """Above the floor, the experiment's own spread governs.

    The spread is the baseline's, measured *before* any arm is compared — the
    pre-registered standard, chosen because it is available before the
    comparison and cannot depend on the arm being judged.
    """
    assert verdicts.materiality_threshold(spread) == spread


def test_the_floor_is_the_boundary_not_a_bracket():
    """At exactly the floor the threshold is the floor, and a delta equal to it
    is not material.

    A delta must *exceed* the threshold, so equality is noise. That keeps the
    rule's promise: a threshold of 0.02 does not fire on a delta of 0.02.
    """
    assert verdicts.materiality_threshold(0.02) == 0.02
    assert verdicts.verdict(0.02, 0.02) == verdicts.WITHIN_SEED_NOISE
    assert verdicts.verdict(0.0201, 0.02) == verdicts.MATERIAL_IMPROVEMENT


# --------------------------------------------------------------------------- #
# The verdicts
# --------------------------------------------------------------------------- #


def test_an_improvement_exceeding_the_spread_is_material():
    assert verdicts.verdict(+0.5, 0.1) == verdicts.MATERIAL_IMPROVEMENT
    assert verdicts.verdict(+0.021, 0.0) == verdicts.MATERIAL_IMPROVEMENT


def test_a_decline_exceeding_the_spread_is_material_harm():
    assert verdicts.verdict(-0.5, 0.1) == verdicts.MATERIAL_HARM
    assert verdicts.verdict(-0.021, 0.0) == verdicts.MATERIAL_HARM


@pytest.mark.parametrize("delta", [0.0, 0.005, -0.005, 0.0199, -0.0199])
def test_a_delta_inside_the_spread_is_noise(delta):
    """Inside the threshold, the delta is indistinguishable from a re-draw of
    the same seeds, which is not a result."""
    assert verdicts.verdict(delta, 0.02) == verdicts.WITHIN_SEED_NOISE


def test_a_delta_between_the_floor_and_the_spread_is_noise():
    """The spread is larger than the floor here, so the spread governs.

    A delta of 0.03 against a spread of 0.1 is well inside the experiment's
    own noise even though it exceeds the floor; reading it as material would
    be using the wrong threshold, and the floor is only a minimum.
    """
    assert verdicts.verdict(0.03, 0.1) == verdicts.WITHIN_SEED_NOISE
    assert verdicts.verdict(0.11, 0.1) == verdicts.MATERIAL_IMPROVEMENT


def test_the_threshold_is_symmetric():
    """Improvement and harm sit the same distance from zero.

    The rule must not read as lenient toward decline, which is how a
    "no worse than before" reading of a small negative delta would otherwise
    survive. The same magnitude, opposite sign, gets the opposite verdict.
    """
    spread = 0.15
    for delta in (0.2, 0.3, 0.4, 1.0):
        v_up = verdicts.verdict(+delta, spread)
        v_down = verdicts.verdict(-delta, spread)
        assert v_up == verdicts.MATERIAL_IMPROVEMENT
        assert v_down == verdicts.MATERIAL_HARM, (
            f"a decline of {delta} against spread {spread} is {v_down!r}, not "
            "material harm — the rule is asymmetric"
        )
    # And symmetric in the noise band.
    for delta in (0.01, 0.05, 0.14):
        assert verdicts.verdict(+delta, spread) == verdicts.WITHIN_SEED_NOISE
        assert verdicts.verdict(-delta, spread) == verdicts.WITHIN_SEED_NOISE


def test_zero_is_noise():
    assert verdicts.verdict(0.0, 0.0) == verdicts.WITHIN_SEED_NOISE


# --------------------------------------------------------------------------- #
# The seed spread
# --------------------------------------------------------------------------- #


def test_the_seed_spread_is_max_minus_min():
    """Not a standard deviation and not a mean-normalised range.

    Max-minus-min is the pre-registered measure, and a script computing it
    differently would be applying a different threshold to the same numbers
    with no visible difference in the output.
    """
    assert verdicts.seed_spread([0.4, 0.1, 0.9, 0.5]) == 0.8
    assert verdicts.seed_spread([0.5, 0.5, 0.5]) == 0.0
    assert verdicts.seed_spread([1.0]) == 0.0


def test_the_seed_spread_is_order_independent():
    """Seeds are an unordered set in the statistics.

    A spread that moved when a script changed its seed ordering would move the
    materiality threshold for no scientific reason.
    """
    vals = [0.2, 0.7, 0.4, 0.9, 0.1]
    assert verdicts.seed_spread(vals) == verdicts.seed_spread(list(reversed(vals)))


def test_the_seed_spread_of_nothing_is_zero():
    """Defined rather than raising.

    An evaluation cell with no steps must not crash the report; it produces a
    zero spread, which the floor turns into the registered minimum, and the
    gate's tolerance is then the honest one.
    """
    assert verdicts.seed_spread([]) == 0.0


def test_the_seed_spread_does_not_collapse_negative_values():
    """The spread is over the values, not over their magnitudes.

    ``delta_1`` is negative at large H, and a spread computed on signed values
    is what the margin's reproducibility actually is.
    """
    assert verdicts.seed_spread([-1.2, -0.4, -2.0]) == 1.6


# --------------------------------------------------------------------------- #
# The delta detail record
# --------------------------------------------------------------------------- #


def test_delta_detail_fixes_the_sign_convention():
    """``delta = observed - baseline``, and positive is an improvement.

    Two measurement scripts with opposite conventions would invert every
    comparison between their records without raising, which is the failure a
    shared sign convention exists to prevent. The field set is fixed so a
    reader's check applies to every record.
    """
    d = verdicts.delta_detail(
        arm="analytic_margin", metric="routing@1",
        baseline=0.10, observed=0.16, baseline_spread=0.02,
    )
    assert d["delta"] == pytest.approx(+0.06)
    assert d["verdict"] == verdicts.MATERIAL_IMPROVEMENT
    assert set(d) == {
        "arm", "metric", "baseline", "observed", "delta",
        "baseline_spread", "threshold", "verdict",
    }
    # The recorded threshold is the one that produced the recorded verdict, so
    # a reader can re-derive the judgement from the record alone.
    assert d["threshold"] == verdicts.materiality_threshold(0.02)
    assert d["threshold"] == max(d["baseline_spread"], verdicts.SPREAD_FLOOR)


def test_delta_detail_applies_the_floor_to_its_own_threshold():
    """The detail record carries the floored threshold, not the raw spread.

    A record that reported the raw spread as its threshold would let a reader
    apply a different rule from the one the script used, silently, by
    computing the materiality from the printed spread.
    """
    d = verdicts.delta_detail(
        arm="x", metric="y", baseline=0.0, observed=0.5, baseline_spread=0.005,
    )
    assert d["threshold"] == verdicts.SPREAD_FLOOR
    assert d["baseline_spread"] == 0.005  # the raw spread is preserved
    assert d["verdict"] == verdicts.MATERIAL_IMPROVEMENT


def test_delta_detail_carries_its_context():
    """The context is the experiment's own tag, not a field the shared module
    has to know about.

    F3 tags a comparison with its candidate count; a different experiment may
    tag one with a different quantity. The shared part is the verdict and the
    threshold, and what is *not* shared is left to the caller rather than
    forced into a field the module would have to maintain.
    """
    d = verdicts.delta_detail(
        arm="epsilon_greedy", metric="recall@16",
        baseline=0.30, observed=0.22, baseline_spread=0.05,
        context={"h": 256, "arm_class": "exploration"},
    )
    assert d["h"] == 256 and d["arm_class"] == "exploration"
    assert d["verdict"] == verdicts.MATERIAL_HARM


def test_delta_detail_without_context_carries_no_extra_fields():
    d = verdicts.delta_detail(
        arm="x", metric="y", baseline=0.0, observed=0.0, baseline_spread=0.0,
    )
    assert "h" not in d
    assert d["verdict"] == verdicts.WITHIN_SEED_NOISE


# --------------------------------------------------------------------------- #
# The scripts use the shared rule, and use it once per delta
# --------------------------------------------------------------------------- #


def test_the_measurement_scripts_do_not_redefine_the_rule():
    """The scripts must call the shared verdict, not their own copy.

    The rule lived in both scripts before the measurement package, and each
    copy had its own floor. What this asserts is that no copy has come back —
    a private ``def _verdict`` or a live ``> 0.02`` comparison in a
    measurement script is the drift reappearing in the place a reviewer is
    least likely to look.

    Comments and docstrings are excluded from the literal check, because the
    floor *should* be explained in prose where a reader meets it; what must
    not come back is a second copy in code, which is what would drift.
    """
    import ast
    import re

    for name in ("measure_surrogate.py", "measure_learn.py",
                 "measure_matched_h.py"):
        path = REPO_ROOT / "scripts" / name
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        # A local reimplementation of the rule itself.
        assert not re.search(r"def\s+_seed_spread\s*\(", text), (
            f"{name} redefines seed_spread; the shared definition in "
            "tac_osm.measurement.verdicts is the one the records must agree with"
        )
        assert not re.search(r"def\s+_verdict\s*\(", text), (
            f"{name} redefines the materiality verdict locally, which is the "
            "drift the shared module exists to remove"
        )
        # The floor as a live numeric literal, anywhere in code. Comments are
        # not code, so this walks the syntax rather than the text: the floor
        # is explained in prose where a reader meets it, and a second copy in
        # code is what would drift.
        literals = {
            float(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
        }
        assert 0.02 not in literals, (
            f"{name} has a live 0.02 literal in code; the floor is a "
            "pre-registered constant and must be read from "
            "verdicts.SPREAD_FLOOR so a change to it is a change to one name"
        )
        # And the shared rule must actually be reached.
        assert "_verdicts." in text or "verdicts." in text, (
            f"{name} does not reach the shared verdict module"
        )


def test_matched_h_records_no_status():
    """MATCHED-001's record carries no verdict, only the comparison.

    Its two pre-committed branches are qualitative — "recovers", "falls as
    steeply as" — which are thresholds a reader applies against published
    numbers, not ones the instrument computes. The instrument reaches the
    shared module for the *spread* (measurement) but never for a *verdict*
    (arbitration), which is the honest division: the record reports the two
    values and their difference, and the reader judges whether it "recovers".
    """
    text = (REPO_ROOT / "scripts" / "measure_matched_h.py").read_text(
        encoding="utf-8")
    # ``seed_spread`` is measurement — the reproducibility of the number.
    assert "_verdicts.seed_spread(" in text, (
        "measure_matched_h.py does not use the shared seed_spread; an inline "
        "max-minus-min is a third copy of the threshold's input"
    )
    # A verdict or a delta detail is arbitration, and this script's rule has
    # no threshold to apply: "recovers" is a comparison against a published
    # row, not a materiality test.
    for arb in ("_verdicts.verdict(", "_verdicts.delta_detail(",
                "_verdicts.materiality_threshold(", "MATERIAL"):
        assert arb not in text, (
            f"measure_matched_h.py reaches {arb}, but its pre-registered rule "
            "is qualitative and has no threshold to apply — the instrument "
            "would be arbitrating a comparison the reader is meant to make"
        )


def test_matched_h_records_the_diagonal_comparison():
    """The record must carry the evidence the two branches read.

    Both branches compare the diagonal against the H=8 transfer row, so that
    comparison is what the record supplies: the two values and their
    difference, per endpoint and population, and no conclusion.
    """
    text = (REPO_ROOT / "scripts" / "measure_matched_h.py").read_text(
        encoding="utf-8")
    assert "h_train_matched" in text and "h_train_8x_transfer" in text, (
        "the MATCHED-001 record does not carry the diagonal-vs-transfer "
        "comparison its decision rule reads"
    )
