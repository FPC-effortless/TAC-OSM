from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.casm_s_adapter import CASMSAdapter, WorkAccounting


def test_execution_work_accounting_contract():
    w = WorkAccounting(3, 17, 17, 6)
    assert w.total == 43


def test_m2_contract_is_preregistered():
    import json
    p = ROOT / "contracts" / "TACOSM-M2-CASM-SELECTIVE-009.json"
    data = json.loads(p.read_text())
    assert data["status"] == "preregistered"
    assert data["external_executor"]["commit"] == "c31554413301e3c9d3e6b3f8c8c6be572a74a748"
    assert data["leakage_rules"]
    assert data["primary_endpoint"]["capability_floor"] == 0.8


def test_router_arms_have_identical_trainable_parameter_count():
    from scripts.run_m2_casm_selective_009 import TwoTowerRouter
    structural = TwoTowerRouter(32, seed=0)
    summary = TwoTowerRouter(32, seed=17)
    assert sum(p.numel() for p in structural.parameters()) == sum(
        p.numel() for p in summary.parameters()
    )


def test_task_address_depends_on_public_examples_not_target_index():
    import hashlib
    from scripts.run_m2_casm_selective_009 import make_task_pool
    task, _ = make_task_pool(19, 32, heldout=False, train_structures=set())
    digest = hashlib.sha256(repr(task.examples).encode("utf-8")).hexdigest()[:16]
    assert task.task_id.endswith(digest)
    assert task.task_id != f"m2:19:32:0:{task.target_index}"


def test_verifier_covers_full_truth_table_complement():
    from scripts.run_m2_casm_selective_009 import make_task_pool
    task, candidates = make_task_pool(23, 32, heldout=False, train_structures=set())
    assert len(task.examples) == 4
    assert len(task.verification_examples) == 12
    target = candidates[task.target_index]
    target_truth = tuple(target.truth_table[k] for k in sorted(target.truth_table))
    assert all(
        tuple(c.truth_table[k] for k in sorted(c.truth_table)) != target_truth
        for c in candidates
        if c is not target
    )
