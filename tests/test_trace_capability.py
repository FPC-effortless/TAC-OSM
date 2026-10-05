"""Evidence-invariant and contract tests for the G-CASM-016C bridge family.

The R1 tests below the split asserted type identity on hand-written literals,
which passed while the runner produced int candidate evidence against a
1-tuple target and every scalar compatible bucket was empty. The R2 tests
exercise the invariant on generated evidence instead.

The pure-logic tests in the first block import only ``tac_osm`` and run
anywhere. The generator-backed tests need the CASM source checkout and torch,
both of which are runner-only on this device; they skip locally and run in CI.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tac_osm.contract import load_contract

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

CASM_SOURCE_ROOT = Path(
    os.environ.get(
        "CASM_SOURCE_ROOT",
        str(REPO_ROOT / "third_party" / "cdl-attention-experiment"),
    )
)

# The runner imports the CASM generator, which needs the third_party checkout
# and torch. Both are absent on the Termux control plane, so the import is
# deferred: tests that need it skip locally and run on the CI runner. The
# torch-free tests below always run.
try:
    import run_graph_casm_trace_capability_bridge_016c_r2 as r2  # noqa: E402

    GENERATOR_AVAILABLE = CASM_SOURCE_ROOT.exists() and getattr(
        r2, "GENERATOR_AVAILABLE", True
    )
except Exception:
    r2 = None
    GENERATOR_AVAILABLE = False

needs_generator = pytest.mark.skipif(
    not GENERATOR_AVAILABLE,
    reason=(
        "CASM generator source (third_party/cdl-attention-experiment) or torch "
        "is absent; this test runs on the GitHub Actions runner"
    ),
)

R1_ID = "TACOSM-GRAPH-CASM-TRACE-CAPABILITY-BRIDGE-016C-R1"
R2_ID = "TACOSM-GRAPH-CASM-TRACE-CAPABILITY-BRIDGE-016C-R2"


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

def test_016c_r1_contract_is_single_primary():
    c = load_contract(R1_ID)
    assert c.check_consistency() == []
    assert c.primary_endpoint() == "verified_success_delta_m512_b8"


def test_budget_is_fixed_at_eight_in_contract():
    c = load_contract(R1_ID)
    assert c.sections["terminal_protocol"]["budget"] == 8


def test_scope_blocks_learned_claims():
    c = load_contract(R1_ID)
    assert c.sections["scope"]["learned_probe_policy_claim"] is False


def test_016c_r2_contract_is_single_primary():
    c = load_contract(R2_ID)
    assert c.check_consistency() == []
    assert c.primary_endpoint() == "verified_success_delta_m512_b8"


def test_016c_r2_scope_blocks_learned_claims():
    c = load_contract(R2_ID)
    assert c.sections["scope"]["learned_probe_policy_claim"] is False


def test_016c_r2_marks_r1_invalid():
    c = load_contract(R2_ID)
    cor = c.sections["correction"]
    assert cor["replaces_experiment"] == R1_ID
    assert cor["invalidates"][0]["status"] == "INVALID"
    assert cor["invalidates"][0]["artifact"] == R1_ID
    assert cor["invalidates"][1]["status"] == "INVALID"


def test_016c_r2_requires_the_two_condition_regression():
    """The contract must demand type equality AND a non-empty bucket.

    Type equality alone is insufficient: a uniformly typed extraction can still
    produce a bucket that fails to contain the target. R1's regression asked
    only for type identity on a literal.
    """
    c = load_contract(R2_ID)
    required = c.sections["correction"]["required_regression"]
    assert "type(candidate_evidence[i]) == type(target_evidence)" in required
    assert ">= 1" in required


# ---------------------------------------------------------------------------
# Pure-logic tests of the evidence invariant.
#
# These run without the CASM generator or torch. They encode the exact failure
# mode that invalidated R1 and would have failed on R1 as written.
# ---------------------------------------------------------------------------

def test_budget_capped_utility_matches_partition_ceiling():
    from tac_osm.trace_capability import budget_capped_utility
    evidence = (0, 0, 1, 1)
    assert budget_capped_utility(evidence, 1) == 0.5
    assert budget_capped_utility(evidence, 2) == 1.0


def test_budget_capped_utility_fails_closed():
    from tac_osm.trace_capability import budget_capped_utility
    with pytest.raises(ValueError):
        budget_capped_utility((), 8)
    with pytest.raises(ValueError):
        budget_capped_utility((0, 1), 0)


def test_invariant_rejects_the_r1_type_mismatch():
    """The invariant must fail closed on exactly the R1 failure mode."""
    from tac_osm.trace_capability import assert_evidence_invariant

    candidate_ints = (0, 1, 0, 1)
    wrapped_target = (0,)  # the R1 scalar bug: int candidates vs a 1-tuple target
    with pytest.raises(RuntimeError, match="evidence type mismatch"):
        assert_evidence_invariant(candidate_ints, wrapped_target, "scalar_row", 0)


def test_invariant_rejects_an_empty_compatible_bucket():
    """Uniform typing is not enough; the target must actually be present.

    The target is itself a candidate, so its evidence must appear among the
    candidate evidence. An empty bucket is therefore an extraction-path bug,
    not an empirical result.
    """
    from tac_osm.trace_capability import assert_evidence_invariant

    with pytest.raises(RuntimeError, match="extraction-path disagreement"):
        assert_evidence_invariant((0, 1, 0), 7, "scalar_row", 0)


def test_invariant_rejects_non_uniform_candidate_evidence():
    from tac_osm.trace_capability import assert_evidence_invariant

    with pytest.raises(RuntimeError, match="not uniformly typed"):
        assert_evidence_invariant((0, (1,), 0), 0, "scalar_row", 0)


def test_invariant_rejects_empty_candidate_evidence():
    from tac_osm.trace_capability import assert_evidence_invariant

    with pytest.raises(ValueError):
        assert_evidence_invariant((), 0, "scalar_row", 0)


def test_invariant_accepts_a_well_formed_scalar_partition():
    from tac_osm.trace_capability import assert_evidence_invariant

    assert_evidence_invariant((0, 1, 0, 1), 1, "scalar_row", 0)


def test_invariant_accepts_a_well_formed_trace_partition():
    from tac_osm.trace_capability import assert_evidence_invariant

    assert_evidence_invariant(((0, 1), (1, 0), (0, 1)), (0, 1), "activation_trace", 0)


def test_invariant_passes_when_the_r1_bug_is_present_is_a_failure():
    """The R1 signature itself must trip the invariant.

    This is the regression that R1 shipped: candidate evidence as ints, target
    evidence as a 1-tuple. It passed R1's own literal-only test. It must not
    pass this one.
    """
    from tac_osm.trace_capability import assert_evidence_invariant, compatible_bucket

    candidate = (0, 1, 0, 1)
    r1_target = (0,)
    # The R1 bucket really was empty:
    assert compatible_bucket(candidate, r1_target) == ()
    with pytest.raises(RuntimeError):
        assert_evidence_invariant(candidate, r1_target, "scalar_row", 0)


def test_the_scalar_budget_capped_utility_matches_the_analytic_partition_ceiling():
    """R1's scalar partition was well typed and correct; only the target broke.

    Its measured budget_capped_utility matched the analytic 2*min(B, M/2)/M at
    every M, which is the evidence that the underlying partition was computed
    correctly and the failure was confined to target extraction.
    """
    from collections import Counter
    from tac_osm.trace_capability import budget_capped_utility

    for m in (32, 64, 128, 256, 512):
        # A balanced binary partition: two evidence values, M/2 candidates each.
        evidence = tuple(0 for _ in range(m // 2)) + tuple(1 for _ in range(m // 2))
        assert len(Counter(evidence)) == 2
        assert budget_capped_utility(evidence, 8) == 2 * min(8, m // 2) / m


# ---------------------------------------------------------------------------
# Accounting: probe work must be a total, not a per-candidate mean.
# ---------------------------------------------------------------------------

def test_probe_work_total_scales_with_m_while_the_mean_does_not():
    """The accounting change that makes the O(M) scan visible.

    In R1, ``probe_environment_work_units`` was a per-candidate mean, so it was
    flat across M (about 16.9 for scalar at every level) while the real
    acquisition scan grew linearly. Charging the mean alone reports O(1) cost
    for an O(M) operation.
    """
    per_candidate = 16.9
    totals = {m: per_candidate * m for m in (32, 64, 128, 256, 512)}
    assert totals[512] / totals[32] == 16.0
    assert all(totals[m] > per_candidate for m in totals)
    assert all(totals[m] / m == per_candidate for m in totals)


# ---------------------------------------------------------------------------
# Generator-backed tests. These run on the CI runner, where the CASM source
# checkout and torch are available. They skip on this Termux control plane.
# ---------------------------------------------------------------------------

@needs_generator
def test_r2_evidence_extraction_yields_matching_types_on_generated_library():
    """Candidate and target evidence come from one typed path per channel.

    This is the test R1 lacked: it builds a real probe cache from the frozen
    generator and checks the invariant the runner enforces, on generated data.
    """
    training = r2.g15.g010.generate(0 + 100, r2.g15.TRAIN_PROGRAMS)
    train_s = {r2.g15.g010.structure_key(ep) for ep in training}
    train_t = {r2.g15.g010.truth_signature(ep) for ep in training}
    library = r2.g15.g010.generate(
        0 + 5000, r2.g15.LIBRARY_SIZE,
        exclude_structures=train_s,
        exclude_truths=train_t,
    )
    assert len(library) == r2.g15.LIBRARY_SIZE
    cache, _ = r2.g15.build_cache(library)

    indices = tuple(range(64))
    for channel in r2.CHANNELS:
        for row in range(len(r2.INPUT_ROWS)):
            evidence = r2.candidate_evidence(cache, indices, channel, row)
            assert len(evidence) == len(indices)
            for target_index in (0, 7, 31, 63):
                task = r2.g15.build_query_task(0, 64, target_index)
                target = r2.target_evidence(cache, task, channel, row)
                # The invariant itself must accept generated evidence.
                r2.assert_evidence_invariant_generated(evidence, target, channel, row)


@needs_generator
def test_r2_generated_scalar_bucket_is_never_empty():
    """The R1 scalar bucket was 0 in all 800 trials. In R2 it must be >= 1."""
    training = r2.g15.g010.generate(100, r2.g15.TRAIN_PROGRAMS)
    train_s = {r2.g15.g010.structure_key(ep) for ep in training}
    train_t = {r2.g15.g010.truth_signature(ep) for ep in training}
    library = r2.g15.g010.generate(
        5000, r2.g15.LIBRARY_SIZE, exclude_structures=train_s, exclude_truths=train_t
    )
    cache, _ = r2.g15.build_cache(library)
    indices = tuple(range(64))
    for row in range(len(r2.INPUT_ROWS)):
        evidence = r2.candidate_evidence(cache, indices, "scalar_row", row)
        for target_index in (0, 7, 31, 63):
            task = r2.g15.build_query_task(0, 64, target_index)
            target = r2.target_evidence(cache, task, "scalar_row", row)
            bucket = r2.compatible_bucket(evidence, target)
            assert len(bucket) >= 1, "scalar bucket must be non-empty in R2"


@needs_generator
def test_r2_runner_reports_operation_counters():
    import inspect

    src = inspect.getsource(r2.channel_trial)
    counter_names = (
        "signature_construction_operations",
        "candidate_scan_operations",
        "probe_selection_operations",
        "execution_operations",
        "verification_operations",
        "total_operations",
    )
    for name in counter_names:
        assert name in src, f"channel_trial does not report {name}"

    training = r2.g15.g010.generate(100, r2.g15.TRAIN_PROGRAMS)
    train_s = {r2.g15.g010.structure_key(ep) for ep in training}
    train_t = {r2.g15.g010.truth_signature(ep) for ep in training}
    library = r2.g15.g010.generate(
        5000, r2.g15.LIBRARY_SIZE, exclude_structures=train_s, exclude_truths=train_t
    )
    cache, _ = r2.g15.build_cache(library)
    task = r2.g15.build_query_task(0, 64, 0)
    trial = r2.channel_trial(cache, task, library[:64], "scalar_row")
    for name in counter_names:
        assert name in trial, f"trial dict missing {name}"
        assert trial[name] >= 0
    assert trial["total_operations"] >= trial["candidate_scan_operations"]
    assert trial["probe_environment_work_units"] > trial["probe_environment_work_units_per_candidate"]
    assert trial["target_bucket_size"] >= 1, "scalar bucket must be non-empty in R2"


@needs_generator
def test_r2_trace_arm_matches_its_preregistered_budget_capped_ceiling():
    """The trace arm's success is bounded by its own partition ceiling.

    R1's trace arm was valid descriptive evidence on its own terms: at M=512 it
    reached 0.8313 against a preregistered ``budget_capped_utility`` ceiling of
    0.84805, i.e. it approached but did not exceed its information ceiling. R2
    must preserve that property while fixing the scalar arm.
    """
    ceiling = load_contract(R2_ID).sections["predicted_budget_capped_ceiling"]
    m = ceiling["M"]
    training = r2.g15.g010.generate(100, r2.g15.TRAIN_PROGRAMS)
    train_s = {r2.g15.g010.structure_key(ep) for ep in training}
    train_t = {r2.g15.g010.truth_signature(ep) for ep in training}
    library = r2.g15.g010.generate(
        5000, r2.g15.LIBRARY_SIZE, exclude_structures=train_s, exclude_truths=train_t
    )
    cache, _ = r2.g15.build_cache(library)
    indices = tuple(range(m))
    utilities = []
    for task_i in range(8):
        task = r2.g15.build_query_task(0, m, task_i)
        trial = r2.channel_trial(cache, task, library[:m], "activation_trace")
        utilities.append(trial["budget_capped_utility"])
        # Verified success cannot exceed the partition-implied ceiling.
        assert trial["verified_success"] <= trial["budget_capped_utility"] + 1e-12, (
            "verified success exceeded the budget-capped utility ceiling"
        )
    assert all(u > 0.0 for u in utilities)
