"""Tests for the model-state integrity gate.

The gate exists because of a specific, documented failure in TACOSM-HS-001:
an evaluation read routing metrics from a router whose weights were never
loaded, so ``w = [0]*n`` scored every candidate uniformly and ``routing@1``
came out at ~1/40. That was interpreted as a finding about the router. It was
a finding about the harness.

These tests pin the gate's behaviour and pin the failure class itself, so a
future measurement that makes the same mistake fails here rather than in the
results table.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm import Candidate, PersistentState, Query  # noqa: E402
from tac_osm.integrity import (  # noqa: E402
    UNTRAINED_HASH,
    Checkpoint,
    IntegrityError,
    assert_trained,
    checkpoint_from_dict,
    checkpoint_to_dict,
    is_untrained,
    manifest_to_dict,
    parameter_hash,
    parameter_norm,
    snapshot_router,
    untrained_hash_for,
)
from tac_osm.router import LearnedRelationalRouter, RouterConfig  # noqa: E402
from tac_osm.state import PersistentStore, StateConfig  # noqa: E402


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #


def test_zero_vector_hashes_to_untrained_hash_at_its_own_width():
    """The manifest's untrained comparison must be exact, not approximate.

    A content hash cannot be width-independent — packing more zeros extends
    the byte string — so the manifest records the untrained hash *of the
    checkpoint's own width* rather than a universal constant, and the
    comparison still holds at every basis size.
    """
    for n in (1, 2, 10, 200):
        assert untrained_hash_for([0.0] * n) == parameter_hash([0.0] * n)
    assert UNTRAINED_HASH == parameter_hash([0.0])


def test_hash_is_stable_across_objects_of_equal_value():
    a = parameter_hash([0.5, -1.0, 0.0])
    b = parameter_hash([0.5, -1.0, 0.0])
    assert a == b


def test_hash_is_not_the_python_object_hash():
    """``hash(tuple)`` is salted per interpreter and cannot identify a run."""
    assert parameter_hash([0.5]) != UNTRAINED_HASH


def test_hash_distinguishes_trained_from_untrained():
    assert parameter_hash([0.0, 0.0, 0.0]) != parameter_hash([0.0, 0.0, 0.1])


def test_hash_ignores_the_sign_of_zero():
    """The analytic weights are zero across whole blocks.

    ``-0.0 == 0.0`` but they differ in bits, so a sign difference would make
    two identical weight vectors hash differently and the gate would flag a
    checkpoint that matches the analytic vector exactly.
    """
    assert parameter_hash([0.0, 1.0]) == parameter_hash([-0.0, 1.0])


def test_hash_depends_on_length():
    """A padded and an unpadded basis must not be interchangeable."""
    assert parameter_hash([0.1]) != parameter_hash([0.1, 0.0])


def test_untrained_detection_is_exact():
    assert is_untrained([0.0, 0.0]) is True
    assert is_untrained([]) is True
    assert is_untrained([0.0, 1e-12]) is False


def test_an_empty_vector_is_untrained_but_is_not_a_checkpoint_basis():
    """The router basis is never empty, so the empty vector is not a state.

    It is nonetheless untrained by the predicate, which is the test the gate
    enforces. The two questions are separate and both are answered here.
    """
    assert is_untrained([]) is True
    assert parameter_hash([]) != UNTRAINED_HASH


def test_norm_is_zero_only_when_untrained():
    assert parameter_norm([0.0, 0.0]) == 0.0
    assert parameter_norm([3.0, 4.0]) == 5.0


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #


def test_checkpoint_records_its_own_hash():
    cp = Checkpoint(
        arm="learned",
        weights=(0.5, -0.5, 0.0),
        hash=parameter_hash([0.5, -0.5, 0.0]),
    )
    assert cp.hash == parameter_hash([0.5, -0.5, 0.0])
    assert cp.untrained is False


def test_checkpoint_rejects_weights_that_do_not_match_its_hash():
    """A checkpoint is a claim about a state; the claim must be true."""
    with pytest.raises(IntegrityError, match="does not match"):
        Checkpoint(
            arm="learned",
            weights=(0.5, 0.0),
            hash=parameter_hash([0.0, 0.0]),
        )


def test_untrained_checkpoint_is_detected_by_the_gate():
    """The constructor permits it; the gate rejects it. Distinct responsibilities."""
    weights = (0.0, 0.0)
    cp = Checkpoint(arm="learned", weights=weights, hash=parameter_hash(weights))
    assert cp.untrained is True
    with pytest.raises(IntegrityError, match="untrained"):
        assert_trained(cp)


def test_a_none_checkpoint_is_a_legible_absence():
    """No learned parameters is not an error — it is recorded, not omitted."""
    assert_trained(None)  # does not raise


def test_a_trained_checkpoint_passes():
    weights = [0.5, -0.5, 0.25]
    cp = Checkpoint(
        arm="learned", weights=tuple(weights), hash=parameter_hash(weights)
    )
    assert_trained(cp)  # does not raise


# --------------------------------------------------------------------------- #
# Snapshotting
# --------------------------------------------------------------------------- #


def test_snapshot_captures_the_router_state():
    router = LearnedRelationalRouter(RouterConfig(dim=4, max_state_slots=2, seed=1))
    router.w = [0.1 * (i % 3) for i in range(router.n)]
    router._updates = 7
    cp = snapshot_router(router)
    assert cp is not None
    assert cp.arm == "learned"
    assert cp.hash == parameter_hash(router.w)
    assert cp.n_updates == 7
    assert cp.norm == pytest.approx(parameter_norm(router.w))
    assert "bias" in cp.signature


def test_snapshot_marks_a_copied_router_as_copied():
    """A sweep must be able to tell a loaded state from one it trained itself.

    The weights are identical either way; only the provenance tag separates
    them, and that is what keeps a frozen-copy sweep from being read as a
    learning curve.
    """
    router = LearnedRelationalRouter(RouterConfig(dim=4, max_state_slots=2, seed=1))
    router.load_weights([0.1] * router.n, source="TACOSM-HS-001:train@8")
    cp = snapshot_router(router)
    assert cp is not None
    assert cp.source.startswith("TACOSM-HS-001")
    assert "copied" not in cp.source


def test_snapshot_of_an_arm_without_parameters_is_none():
    from tac_osm.router import RandomRouter

    assert snapshot_router(RandomRouter()) is None
    assert_trained(snapshot_router(RandomRouter()))  # None is not a failure


def test_snapshot_does_not_mutate_the_router():
    router = LearnedRelationalRouter(RouterConfig(dim=4, max_state_slots=2))
    before = list(router.w)
    snapshot_router(router)
    assert router.w == before


# --------------------------------------------------------------------------- #
# The HS-001 failure class, pinned
# --------------------------------------------------------------------------- #
#
# The exact route by which a false routing conclusion was produced: copy
# weights from a trained router into an evaluation router, then never actually
# copy them. The evaluation router keeps its own zeros. Its outputs are still
# smooth, bounded, and plausible-looking, so nothing downstream objects.


def _store() -> PersistentState:
    return PersistentStore(StateConfig(seed=0, n_slots=4))


def _query() -> Query:
    return Query(text="0 1 1 0\taddr", context=(1, 0, 1, 0))


def _candidates(n: int = 4) -> list[Candidate]:
    return [Candidate(key=f"c{i}", descriptor=tuple((i >> j) & 1 for j in range(4)))
            for i in range(n)]


def test_the_zero_weight_router_scores_uniformly():
    """This is what made the failure invisible: the output looks reasonable."""
    router = LearnedRelationalRouter(RouterConfig(dim=4, max_state_slots=2))
    scores = router.score(_query(), _store(), _candidates())
    assert all(s == scores[0] for s in scores)
    assert all(w == 0.0 for w in router.w)
    # A uniform scorer's decision carries no information at all.
    decision = router.route(_query(), _store(), _candidates())
    assert max(decision.scores) == pytest.approx(min(decision.scores))


def test_the_zero_weight_router_is_caught_before_evaluation():
    """The gate that the HS-001 sweep lacked."""
    router = LearnedRelationalRouter(RouterConfig(dim=4, max_state_slots=2))
    cp = snapshot_router(router)
    with pytest.raises(IntegrityError, match="untrained"):
        assert_trained(cp, context="TACOSM-HS-001:H=32")


def test_score_uses_the_loop_own_scoring_function():
    """The ranking a measurement reports must be the ranking the loop uses."""
    router = LearnedRelationalRouter(RouterConfig(dim=4, max_state_slots=2))
    router.load_weights([1.0] * router.n)
    query, store, cands = _query(), _store(), _candidates()

    from_route = list(router.route(query, store, cands).scores)
    from_score = router.score(query, store, cands)

    # The softmax preserves the ranking of the linear scores, so the two agree
    # in order even though they differ in scale.
    def rank(seq):
        return sorted(range(len(seq)), key=lambda i: -seq[i])

    assert rank(from_route) == rank(from_score)


def test_load_weights_rejects_a_mismatched_basis():
    router = LearnedRelationalRouter(RouterConfig(dim=4, max_state_slots=2))
    with pytest.raises(ValueError, match="cannot load"):
        router.load_weights([0.1] * (router.n + 1))


def test_load_weights_reproduces_the_scored_ranking():
    """A copied router must score exactly as the trained one did."""
    trained = LearnedRelationalRouter(RouterConfig(dim=4, max_state_slots=2, seed=3))
    trained.w = [0.05 * (i % 5 - 2) for i in range(trained.n)]
    query, store, cands = _query(), _store(), _candidates()

    fresh = LearnedRelationalRouter(RouterConfig(dim=4, max_state_slots=2, seed=99))
    fresh.load_weights(trained.w)
    assert fresh.score(query, store, cands) == pytest.approx(
        trained.score(query, store, cands)
    )


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def test_manifest_records_the_comparison_value():
    """A reader must be able to verify the gate ran, not just be told it did."""
    weights = (0.5, 0.25)
    cp = Checkpoint(arm="learned", weights=weights, hash=parameter_hash(weights))
    m = manifest_to_dict(experiment_id="X-001", arm="learned", checkpoint=cp)
    assert m["model_state"]["parameter_hash"] == cp.hash
    assert m["model_state"]["untrained_hash"] == untrained_hash_for(weights)
    assert m["model_state"]["parameter_hash"] != m["model_state"]["untrained_hash"]
    assert m["model_state"]["gate"] == "passed"


def test_manifest_flags_a_failed_gate():
    weights = (0.0, 0.0)
    cp = Checkpoint(arm="learned", weights=weights, hash=parameter_hash(weights))
    m = manifest_to_dict(experiment_id="X-001", arm="learned", checkpoint=cp)
    assert m["model_state"]["gate"] == "FAILED"
    assert m["model_state"]["untrained"] is True
    assert m["model_state"]["untrained_hash"] == m["model_state"]["parameter_hash"]


def test_manifest_names_an_arm_with_no_parameters():
    m = manifest_to_dict(experiment_id="X-001", arm="random", checkpoint=None)
    assert m["model_state"]["parameter_hash"] is None
    assert m["model_state"]["gate"] == "passed"
    assert "no learned parameters" in m["model_state"]["reason"]


def test_round_trip_preserves_the_checkpoint():
    cp = Checkpoint(
        arm="learned",
        weights=(0.5, 0.25, 0.0),
        hash=parameter_hash([0.5, 0.25, 0.0]),
        n_updates=12,
        signature={"bias": 1.0},
    )
    data = checkpoint_to_dict(cp)
    back = checkpoint_from_dict(data)
    assert back is not None
    assert back.hash == cp.hash
    assert back.n_updates == 12
    assert back.signature == {"bias": 1.0}


def test_round_trip_preserves_the_absence():
    back = checkpoint_from_dict(checkpoint_to_dict(None))
    assert back is None


# --------------------------------------------------------------------------- #
# End to end, on the real loop
# --------------------------------------------------------------------------- #


def test_a_trained_model_passes_and_a_fresh_one_fails():
    """The gate as the loop actually uses it, not just on a hand-built vector."""
    from tac_osm.ablation import AblationConfig, RouterSwitch
    from tac_osm.builder import build_model

    bm = build_model(
        AblationConfig(seed=0, router=RouterSwitch(type="learned")),
        n_steps=60,
    )
    before = snapshot_router(bm.model.router)
    assert before is not None
    assert before.untrained is True  # not trained yet — nothing hidden
    # An untrained state may exist; what the gate refuses is an *evaluation*
    # that would consume one.
    with pytest.raises(IntegrityError, match="untrained"):
        assert_trained(before, context="pre-training evaluation")


def test_a_trained_model_passes_the_gate():
    from tac_osm.ablation import AblationConfig, RouterSwitch
    from tac_osm.builder import build_model

    bm = build_model(
        AblationConfig(seed=0, router=RouterSwitch(type="learned")),
        n_steps=200,
    )
    bm.model.run()
    after = snapshot_router(bm.model.router)
    assert after is not None
    assert after.n_updates > 0
    assert not after.untrained
    assert_trained(after)  # a trained router is exactly what should pass


# --------------------------------------------------------------------------- #
# The frozen baseline's provenance row
# --------------------------------------------------------------------------- #
#
# The frozen baseline records the size of the suite at the time it was frozen
# (327) alongside the current size, because a frozen table and a growing suite
# disagree and a reader is entitled to both numbers at once. Written by hand,
# the second number rots on every commit that adds a test — it was stale
# within one commit of being written, which is the same failure class this
# module exists for: a number that looks trustworthy and is not.
#
# So the doc carries a placeholder and `scripts/sync_test_count.py` substitutes
# the live count. This test is the gate that keeps the two in step, and it is
# the one that fails when a test is added and the doc is not refreshed —
# rather than a reader discovering the discrepancy later against a frozen
# table.


def test_the_frozen_baseline_reports_the_current_test_count():
    """The one mutable number in the frozen doc, kept mutable by a script.

    Everything else in `docs/TACOSM-BASELINE-001.md` is frozen; this row's
    second number is the exception, and an exception that is not pinned is the
    exception that silently stops being maintained.
    """
    import re
    import subprocess

    repo_root = Path(__file__).resolve().parent.parent
    doc = repo_root / "docs" / "TACOSM-BASELINE-001.md"
    text = doc.read_text(encoding="utf-8")
    assert "<!--TESTCOUNT-->" not in text, (
        "the frozen baseline carries the placeholder rather than a count; "
        "run `python scripts/sync_test_count.py`"
    )
    # The frozen count is untouched: 327, and only 327.
    m = re.search(r"test count @ freeze \| (\d+) \(current", text)
    assert m, "the 'test count @ freeze' row is missing from the provenance table"
    assert m.group(1) == "327", (
        "the frozen count changed; the freeze is what later deltas compare "
        f"against, and it was 327 (found {m.group(1)!r})"
    )

    r = subprocess.run(
        [sys.executable, "scripts/sync_test_count.py", "--check"],
        cwd=repo_root, capture_output=True, text=True,
    )
    assert r.returncode == 0, (
        "the canonical test-count gate failed:\n"
        f"{r.stdout}\n{r.stderr}"
    )
