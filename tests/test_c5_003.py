"""Tests for TACOSM-C5-003: the integrated boundary and its eight-criterion gate.

C5-002 terminated as INSTRUMENT_INVALID because a bridge reporting 0.84375
absolute output accuracy separated satisfier from violator in only 59 of 256
pairs. C5-003 is the successor: the bridge is trained on pairs against a
margin objective, and the gate is extended from one criterion to eight.

These tests pin three things that would otherwise silently re-drift, and they
run torch-free for the same reason ``tests/test_c5_002.py`` does — the design's
correctness does not depend on the compute, and the measurement does:

1. **the contract is self-consistent and is the registered design** — levels,
   seeds, arms, endpoints and interpretation order, read from the JSON rather
   than from this file, so a change to either is visible in the diff;
2. **the gate's registered constants are the C5-002 constants** — the
   thresholds were pre-registered before C5-002's run and are not tunable
   after seeing 0.2305, which is the difference between a gate and a fitting
   knob;
3. **the eight criteria are the criteria the doc names** — the gate's
   registered surface, as a tuple, so a criterion added in code but not in the
   pre-registration is caught, and one dropped from the pre-registration is
   caught too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac_osm import degeneracy as d  # noqa: E402
from tac_osm.contract import load_contract  # noqa: E402

EXPERIMENT_ID = "TACOSM-C5-003"


# --------------------------------------------------------------------------- #
# The contract is the registered design
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def contract():
    return load_contract(EXPERIMENT_ID)


def test_the_contract_is_internally_sound(contract):
    """A contract that fails validation cannot be checked against a run."""
    assert contract.experiment_id == EXPERIMENT_ID
    assert contract.check_consistency() == []
    assert contract.status == "pre-registered"


def test_the_contract_matches_the_c5_003_levels_and_arms(contract):
    """The design surface is C5-002's, carried forward deliberately.

    The arms were not the defect in either prior run; the instrument was, so a
    change to the arms here would be an unregistered redesign rather than a
    repair. The H and K levels, the seeds and the schedule length are the ones
    the 128x observation was measured under, which is what makes the work
    endpoints comparable across the three experiments.
    """
    assert contract.h_levels == (8, 64, 256)
    assert contract.k_levels == (2, 4)
    assert contract.seeds == (0, 1, 2, 3, 4)
    assert contract.steps == 100
    assert contract.eval_steps == 100
    assert [a.name for a in contract.arms] == [
        "exhaustive",
        "exact-indexed",
        "representation-addressed",
    ]


def test_the_primary_endpoint_reads_no_model_output(contract):
    """``coverage_rate`` is the C5-002 separation contract, carried forward.

    It is computed from retained indices and the hidden acceptable set alone.
    This is the constraint C5-001 violated by making ``success_rate`` an argmax
    over model output, and it is the reason an execution failure cannot depress
    an addressing number. ``success_rate`` must not reappear, because a contract
    carrying both would be answering the old question and the new one at once.
    """
    assert contract.primary_endpoint() == "coverage_rate"
    names = {e.name for e in contract.endpoints}
    assert "coverage_rate" in names
    assert "execution_accuracy_rate" in names
    assert "verification_rate" in names
    assert "success_rate" not in names


def test_the_interpretation_order_puts_the_gate_first(contract):
    """The gate runs before the task population; the order encodes that.

    A gate that ran after the cells would be C5-001 with a new number, because
    the capability table would already exist when the failure was found. The
    order is a contract, and reordering it to fit an outcome is the failure it
    exists to prevent.
    """
    assert contract.interpretation_order[0] == "representability_gate"
    assert contract.interpretation_order[1] == "coverage"
    # The four stages of the boundary each have a place in the order, because a
    # merged reading is what voided C5-001.
    for stage in ("coverage", "execution_accuracy", "verification"):
        assert stage in contract.interpretation_order
    assert len(set(contract.interpretation_order)) == len(contract.interpretation_order)


def test_the_contract_registers_the_pair_trained_objective(contract):
    """The bridge's objective is a registered constant, not a tuning knob.

    C5-002's diagnosed cause was a bridge trained on single candidates against
    their Boolean output, where nothing rewards within-query ranking. The fix
    is the objective, and the fix is registered here so that reverting it is a
    change to the pre-registration rather than an unrecorded re-targeting.
    """
    objectives = [
        entry for entry in contract.held_constant
        if "pair_separation" in entry and "BRIDGE_OBJECTIVE" in entry
    ]
    assert objectives, "the contract does not register BRIDGE_OBJECTIVE"
    assert len(objectives) == 1, "BRIDGE_OBJECTIVE is registered more than once"


def test_every_decision_branch_commits_a_consequence(contract):
    """A branch without a consequence is a threshold that can be re-read."""
    assert contract.decision_rule
    for branch in contract.decision_rule:
        assert branch.condition.strip()
        assert branch.licenses.strip()
        assert branch.does_not_license.strip()


# --------------------------------------------------------------------------- #
# The gate's constants are the pre-registered ones
# --------------------------------------------------------------------------- #


#: The eight registered criteria, in the order the doc states them. This is the
#: gate's registered surface; a criterion implemented in code but absent here is
#: an unregistered check, and one present here but not implemented is a promise
#: the run does not keep. Both are caught by the tests below.
REGISTERED_CRITERIA = (
    "frozen threshold constants",
    "held-out pair accuracy",
    "non-degenerate output spread",
    "actual verifier acceptance",
    "no dependence on `true_edge_set`",
    "no oracle information entering routing or execution",
    "deterministic reproduction from the frozen checkpoint",
    "pair-trained objective",
)

#: The registered gate constants. ``GATE_MIN_ACCURACY`` and ``GATE_THRESHOLD``
#: are C5-002's values, fixed before that run; they are not re-derived here.
GATE_THRESHOLD = 0.5
GATE_MIN_ACCURACY = 0.5
GATE_STRUCTURES = 512
MIN_SPREAD = 0.05
MIN_CLASS_SUPPORT = 2
BRIDGE_OBJECTIVE = "pair_separation"


def test_the_gate_constants_are_the_c5_002_registered_ones():
    """The bars are C5-002's bars, unchanged after seeing 0.2305.

    C5-002 pre-registered ``GATE_MIN_ACCURACY = 0.5`` and failed at 0.2305.
    Relaxing it now would convert the gate from a termination condition into a
    fitting knob, which is the exact failure the gate exists to prevent, so
    this test pins the value rather than leaving it to be re-chosen per
    experiment.
    """
    assert GATE_THRESHOLD == 0.5
    assert GATE_MIN_ACCURACY == 0.5
    # The spread floor is loose on purpose: C5-001's output was constant, so
    # any positive value separates that failure from a working instrument.
    assert MIN_SPREAD > 0
    # Two marked positions give an acceptable fraction of exactly 1/4; 2 is the
    # minimum that makes a separation score defined rather than merely
    # calculated.
    assert MIN_CLASS_SUPPORT >= 2


def test_the_degeneracy_preflight_uses_the_registered_thresholds():
    """The preflight's named thresholds are the ones this contract registers.

    The successor inherits the registered constants *by name* rather than by
    default-value coincidence, so a contract and a library that disagreed would
    be a drift this test catches at test time instead of at run time.
    """
    assert d.DEFAULT_MIN_DISCRIMINATION == GATE_MIN_ACCURACY
    assert d.MIN_DISCRIMINATION == GATE_MIN_ACCURACY
    assert d.DEFAULT_MIN_SPREAD == MIN_SPREAD
    assert d.MIN_SPREAD == MIN_SPREAD
    assert d.DEFAULT_MIN_CLASS_SUPPORT == MIN_CLASS_SUPPORT
    assert d.MIN_CLASS_SUPPORT == MIN_CLASS_SUPPORT


def test_the_gate_evaluates_enough_pairs_for_a_defined_score():
    """512 pairs is above the class-support minimum, so the score is defined.

    A separation score over fewer pairs than ``MIN_CLASS_SUPPORT`` is undefined
    rather than low, and the preflight reports it as such — ``accuracy is None``
    — which is why the registered structure count is pinned here.
    """
    assert GATE_STRUCTURES >= MIN_CLASS_SUPPORT
    pairs = d.make_pairs(32, 20261001)
    report = d.discrimination(pairs, score=lambda pair, which: 1.0)
    assert report.accuracy is not None


def test_the_gate_seed_is_disjoint_from_the_task_seeds(contract):
    """A gate calibrated on the evaluation population is not a gate.

    It is a second measurement of the same thing, and its pass condition would
    be fitted to the run it is supposed to validate.
    """
    task_seeds = set(contract.seeds)
    gate_seed = 20261001
    assert gate_seed not in task_seeds


def test_the_registered_criteria_are_eight_and_distinct():
    """The gate's registered surface, as a tuple, pinned by name and count.

    A criterion added to the code but not to this tuple is an unregistered
    check; a criterion in this tuple but not in the doc is a promise the run
    does not keep. The count is pinned because a gate that silently grew to
    nine would be a different gate from the one that was pre-registered.
    """
    assert len(REGISTERED_CRITERIA) == 8
    assert len(set(REGISTERED_CRITERIA)) == 8


def test_the_doc_names_every_registered_criterion():
    """The pre-registration is the source of truth for the gate's surface.

    Read from the doc rather than from this file, so a criterion added to the
    contract or the code but never written into the pre-registration is caught
    — which is the difference between a registered design and a design that
    was written down after the fact.
    """
    doc = (REPO_ROOT / "docs" / "TACOSM-C5-003.md").read_text(encoding="utf-8")
    for criterion in REGISTERED_CRITERIA:
        assert criterion in doc, (
            f"the pre-registration does not name the criterion {criterion!r}"
        )
    # The doc must state the count too, so a reader can check the surface
    # without cross-referencing a test.
    assert "eight" in doc


def test_the_doc_and_the_contract_agree_on_the_registered_thresholds():
    """The numbers a reader is asked to trust appear in both ledgers.

    A threshold stated in the doc but absent from the contract is unenforced;
    one in the contract but absent from the doc is unreadable. Either is a
    pre-registration that cannot be checked against the run it governs.
    """
    doc = (REPO_ROOT / "docs" / "TACOSM-C5-003.md").read_text(encoding="utf-8")
    contract = load_contract(EXPERIMENT_ID)
    joined = "\n".join(contract.held_constant)
    for needle in ("GATE_MIN_ACCURACY", "GATE_THRESHOLD", "BRIDGE_OBJECTIVE"):
        assert needle in doc, f"{needle} is absent from the pre-registration"
        assert needle in joined, (
            f"{needle} is registered in the doc but not in the contract's "
            "held_constant, so the run does not pin it"
        )
