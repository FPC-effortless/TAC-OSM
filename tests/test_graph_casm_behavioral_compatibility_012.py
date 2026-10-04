from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASM_ROOT = ROOT / "third_party" / "cdl-attention-experiment"
if not CASM_ROOT.exists():
    pytest.skip("pinned external generator is not checked out", allow_module_level=True)

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SPEC = importlib.util.spec_from_file_location(
    "graph_casm_behavioral_compatibility_012",
    ROOT / "scripts" / "run_graph_casm_behavioral_compatibility_012.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_build_task_has_nonempty_verifier_for_registered_support_sizes():
    training = MODULE.g010.generate(9010, MODULE.TRAIN_PROGRAMS)
    train_s = {MODULE.g010.structure_key(ep) for ep in training}
    train_t = {MODULE.g010.truth_signature(ep) for ep in training}
    target = MODULE.g010.build_manifest(
        9011, (32,), 1, exclude_structures=train_s, exclude_truths=train_t
    )[(32, 0)]
    for size in MODULE.SUPPORT_SIZES:
        task, candidates = MODULE.build_task(
            9012, 32, target, train_s, train_t, 0, size, heldout=True
        )
        assert len(task.support) == size
        assert len(task.verify) == 16 - size
        assert task.target_index < len(candidates)
        assert task.target_index in MODULE.exact_support_consistent(
            candidates, task.support
        )


def test_fixed_budget_and_exhaustive_controls_do_not_early_stop():
    programs = MODULE.g010.generate(9013, 8)
    target = programs[0]
    other = next(
        ep for ep in programs[1:]
        if MODULE.g010.truth_signature(ep) != MODULE.g010.truth_signature(target)
    )
    verify = tuple(
        (tuple(bits), int(target.truth_table[bits]))
        for bits in MODULE.INPUT_ROWS
    )
    ok_fixed, work_fixed, _, used_fixed = MODULE.execute_candidates(
        [target, other], [0, 1], verify, stop_on_success=False
    )
    ok_adapt, work_adapt, _, used_adapt = MODULE.execute_candidates(
        [target, other], [0, 1], verify, stop_on_success=True
    )
    assert ok_fixed and ok_adapt
    assert used_fixed == 2
    assert used_adapt == 1
    assert work_fixed >= work_adapt


def test_candidate_order_preserves_behavioral_scores_by_candidate_identity():
    from tac_osm.behavioral_compatibility_router import (
        BehavioralCompatibilityRouter,
        BehavioralRouterConfig,
    )

    model = BehavioralCompatibilityRouter(
        BehavioralRouterConfig(seed=4)
    )
    candidates = MODULE.g010.generate(9014, 8)
    support = tuple(
        (tuple(k), int(candidates[0].truth_table[k]))
        for k in MODULE.INPUT_ROWS[:4]
    )
    order1, scores1, _ = MODULE.rank_candidates(model, candidates, support, "graph")

    permutation = [3, 0, 7, 2, 5, 1, 6, 4]
    permuted = [candidates[i] for i in permutation]
    order2, scores2, _ = MODULE.rank_candidates(model, permuted, support, "graph")

    original_by_key = {
        MODULE.g010.structure_key(candidates[i]): scores1[i] for i in range(len(candidates))
    }
    permuted_by_key = {
        MODULE.g010.structure_key(permuted[i]): scores2[i] for i in range(len(permuted))
    }
    assert original_by_key == permuted_by_key

    ranked_keys1 = [MODULE.g010.structure_key(candidates[i]) for i in order1]
    ranked_keys2 = [MODULE.g010.structure_key(permuted[i]) for i in order2]
    assert ranked_keys1 == ranked_keys2


def test_target_index_metadata_cannot_change_router_scores():
    from dataclasses import replace
    from tac_osm.behavioral_compatibility_router import (
        BehavioralCompatibilityRouter,
        BehavioralRouterConfig,
    )

    model = BehavioralCompatibilityRouter(
        BehavioralRouterConfig(seed=6)
    )
    training = MODULE.g010.generate(9016, MODULE.TRAIN_PROGRAMS)
    train_s = {MODULE.g010.structure_key(ep) for ep in training}
    train_t = {MODULE.g010.truth_signature(ep) for ep in training}
    target = MODULE.g010.build_manifest(
        9017, (32,), 1, exclude_structures=train_s, exclude_truths=train_t
    )[(32, 0)]
    task, candidates = MODULE.build_task(
        9018, 32, target, train_s, train_t, 0, 4, heldout=True
    )
    altered = replace(
        task,
        target_index=(task.target_index + 1) % len(candidates),
    )
    _, scores_a, _ = MODULE.rank_candidates(model, candidates, task.support, "graph")
    _, scores_b, _ = MODULE.rank_candidates(model, candidates, altered.support, "graph")
    assert scores_a == scores_b


def test_support_label_permutation_changes_the_query_representation():
    from tac_osm.behavioral_compatibility_router import (
        BehavioralCompatibilityRouter,
        BehavioralRouterConfig,
    )

    model = BehavioralCompatibilityRouter(
        BehavioralRouterConfig(seed=5)
    )
    candidates = MODULE.g010.generate(9015, 8)
    support = tuple(
        (tuple(k), int(candidates[0].truth_table[k]))
        for k in MODULE.INPUT_ROWS[:4]
    )
    permuted_support = tuple(
        (bits, y) for (bits, _), (_, y) in zip(
            support, tuple(reversed(support))
        )
    )
    _, scores1, _ = MODULE.rank_candidates(model, candidates, support, "graph")
    _, scores2, _ = MODULE.rank_candidates(
        model, candidates, permuted_support, "graph"
    )
    assert any(a != b for a, b in zip(scores1, scores2))
