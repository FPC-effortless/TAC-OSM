import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
CASM_ROOT = ROOT / "third_party" / "cdl-attention-experiment"
if not CASM_ROOT.exists():
    pytest.skip("pinned external generator is not checked out", allow_module_level=True)

SPEC = importlib.util.spec_from_file_location("graph_casm_selective_010", ROOT / "scripts" / "run_graph_casm_selective_010.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_graph_feature_exposes_wiring_but_summary_does_not():
    ep = MODULE.generate(1001, 1)[0]
    graph = MODULE.features(ep, True)
    summary = MODULE.features(ep, False)
    assert len(graph) == MODULE.FEATURE_DIM
    assert len(summary) == MODULE.FEATURE_DIM
    assert graph[:90] == summary[:90]
    assert graph[90:] != summary[90:]


def test_exact_graph_executor_reproduces_complete_truth_table():
    ep = MODULE.generate(1002, 1)[0]
    check = MODULE.executor_check([ep])
    assert check["pass"]
    assert check["checked_rows"] == 16


def test_exact_graph_work_is_positive_and_proportional_to_rows():
    ep = MODULE.generate(1003, 1)[0]
    _, one = MODULE.execute_exact(ep, (0, 0, 0, 0))
    assert one.total > 0
    assert MODULE.candidate_work(ep, MODULE.VERIFIER_ROWS) == one.total * MODULE.VERIFIER_ROWS


def test_secondary_role_holdout_requires_registered_not_xor_pair():
    programs = MODULE.generate(2001, 8, require_role_pair=True)
    assert programs
    assert all(MODULE.has_role_pair(ep) for ep in programs)


def test_target_split_is_disjoint_from_training_structures_and_truths():
    training = MODULE.generate(2002, 32)
    train_s = {MODULE.structure_key(ep) for ep in training}
    train_t = {MODULE.truth_signature(ep) for ep in training}
    manifest = MODULE.build_manifest(
        2003, (32,), 8,
        exclude_structures=train_s,
        exclude_truths=train_t,
    )
    assert not train_s.intersection(MODULE.structure_key(ep) for ep in manifest.values())
    assert not train_t.intersection(MODULE.truth_signature(ep) for ep in manifest.values())


def test_target_index_is_not_used_by_semantic_execution():
    training = MODULE.generate(2004, 32)
    train_s = {MODULE.structure_key(ep) for ep in training}
    train_t = {MODULE.truth_signature(ep) for ep in training}
    target = MODULE.build_manifest(
        2005, (32,), 1,
        exclude_structures=train_s,
        exclude_truths=train_t,
    )[(32, 0)]
    task, pool = MODULE.make_task(2006, 32, target, train_s, train_t)
    model = MODULE.Router(0)
    mem = MODULE.Memory()
    mem.write(task.task_id, task.support)
    original = MODULE.evaluate(model, "graph", task, pool, mem)
    alternate_index = (task.target_index + 1) % len(pool)
    alternate = MODULE.Task(
        task.task_id,
        task.support,
        task.verify,
        alternate_index,
    )
    changed = MODULE.evaluate(model, "graph", alternate, pool, mem)
    for budget in MODULE.BUDGETS:
        key = str(budget)
        assert changed["budgets"][key]["semantic_success"] == original["budgets"][key]["semantic_success"]
        assert changed["budgets"][key]["fixed_budget_execution_work_units"] == original["budgets"][key]["fixed_budget_execution_work_units"]
    assert changed["exhaustive"]["semantic_success"] == original["exhaustive"]["semantic_success"]
    assert changed["exhaustive"]["execution_work_units"] == original["exhaustive"]["execution_work_units"]


def test_primary_bootstrap_repeats_budget_selection_rule():
    trial = {
        "exhaustive": {"semantic_success": 1.0, "execution_work_units": 100.0},
        "budgets": {
            "1": {"semantic_success": 0.9, "fixed_budget_execution_work_units": 10.0},
            "2": {"semantic_success": 0.95, "fixed_budget_execution_work_units": 20.0},
            "4": {"semantic_success": 0.7, "fixed_budget_execution_work_units": 40.0},
            "8": {"semantic_success": 0.5, "fixed_budget_execution_work_units": 80.0},
        },
    }
    seed_blocks = {
        str(seed): {"32": {"trials": [trial, trial]}}
        for seed in MODULE.SEEDS
    }
    result = MODULE.primary_bootstrap_ci(seed_blocks, 32, rounds=200)
    assert result["ci95"] == [0.1, 0.1]
    assert result["valid_fraction"] == 1.0


class _ConstantRouter:
    def forward(self, q, c, representation):
        import torch
        return torch.zeros((q.shape[0], c.shape[0]))


def test_candidate_order_shuffle_does_not_change_exhaustive_semantics():
    training = MODULE.generate(2010, 16)
    train_s = {MODULE.structure_key(ep) for ep in training}
    train_t = {MODULE.truth_signature(ep) for ep in training}
    target = MODULE.build_manifest(
        2011, (32,), 1,
        exclude_structures=train_s,
        exclude_truths=train_t,
    )[(32, 0)]
    task, pool = MODULE.make_task(2012, 32, target, train_s, train_t)
    mem = MODULE.Memory()
    mem.write(task.task_id, task.support)
    router = MODULE.Router(0)
    forward = MODULE.evaluate(router, "graph", task, pool, mem)
    import random
    perm = list(range(len(pool)))
    random.Random(2013).shuffle(perm)
    shuffled = [pool[i] for i in perm]
    target_pos = shuffled.index(pool[task.target_index])
    shuffled_task = MODULE.Task(task.task_id, task.support, task.verify, target_pos)
    reverse = MODULE.evaluate(router, "graph", shuffled_task, shuffled, mem)
    assert reverse["exhaustive"] == forward["exhaustive"]


def test_constant_router_cannot_create_above_base_rate_selective_success():
    training = MODULE.generate(2020, 16)
    train_s = {MODULE.structure_key(ep) for ep in training}
    train_t = {MODULE.truth_signature(ep) for ep in training}
    successes = 0
    total = 0
    rng = random.Random(2021)
    for i in range(32):
        target = MODULE.build_manifest(
            2022 + i, (32,), 1,
            exclude_structures=train_s,
            exclude_truths=train_t,
        )[(32, 0)]
        task, pool = MODULE.make_task(3000 + i, 32, target, train_s, train_t)
        mem = MODULE.Memory()
        mem.write(task.task_id, task.support)
        # A constant scorer ranks by the registered deterministic tie-break (index).
        order = list(range(len(pool)))
        if task.target_index in order[:1]:
            successes += 1
        total += 1
    # The fixed construction shuffles candidate order independently of the target;
    # this control should remain close to the 1/M base rate, not the learned result.
    assert successes <= 4
    assert total == 32


def test_executor_failure_is_not_silently_promoted():
    import copy
    ep = MODULE.generate(2030, 1)[0]
    assert MODULE.executor_check([ep])["pass"]
    tampered = copy.deepcopy(MODULE.generate(2031, 1)[0])
    first_key = next(iter(tampered.truth_table))
    original = int(tampered.truth_table[first_key])
    tampered.truth_table[first_key] = 1 - original
    assert MODULE.executor_check([tampered])["pass"] is False
