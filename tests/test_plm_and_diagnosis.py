from pathlib import Path
import inspect
import json

from tac_osm.integrated_e2e_007_benchmark import (
    DEV_COMBOS,
    E2E006_HELDOUT,
    HELDOUT,
    SEALED_E2E005,
)
from scripts import run_plm_e2e_and_diag as diag

ROOT = Path(__file__).resolve().parents[1]


def test_contract_is_preregistered_and_fixed():
    contract = json.loads(
        (ROOT / "contracts" / "TACOSM-PLM-AND-DIAG-001.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["status"] == "pre-registered"
    assert contract["seeds"] == [0, 1, 2, 3, 4]
    assert contract["steps"] == 300
    assert contract["eval_steps"] == 200
    assert contract["protocol"]["hidden_dim"] == 64
    assert contract["protocol"]["state_write_mode"] == "residual_linear"


def test_diagnostic_training_excludes_all_prior_confirmatory_sets():
    excluded = set(SEALED_E2E005) | set(E2E006_HELDOUT) | set(HELDOUT) | set(DEV_COMBOS)
    assert excluded.isdisjoint(set(diag.TRAIN_COMBOS))
    assert ("and", 4, 11) not in diag.TRAIN_COMBOS
    assert ("and", 5, 9) not in diag.TRAIN_COMBOS


def test_development_queries_are_all_and_and_are_not_e2e007():
    source = inspect.getsource(diag.sample_dev)
    assert "and_combos" in source
    assert "E2E007_HELDOUT" not in source
    assert "0.9350" not in source


def test_dispatch_check_matches_fixed_and_formula():
    source = inspect.getsource(diag.evaluate)
    assert "op_idx = OPS.index("and")" in source
    assert "expected_action = bit_probs[i] * bit_probs[j]" in source
    assert "and_dispatch_mismatches" in source


def test_no_confirmatory_outcome_is_used_for_selection():
    source = inspect.getsource(diag.run) + inspect.getsource(diag.train_seed)
    assert "e2e007_test_outcomes_not_used" in source
    assert "0.9350" not in source
    assert "E2E007_HELDOUT" not in source


def test_output_is_marked_development_only():
    source = inspect.getsource(diag.run)
    assert '"status": "development"' in source
    assert '"no confirmatory claim"' in source


def test_workflow_exists_and_references_exact_runner():
    wf = (ROOT / ".github" / "workflows" / "plm-e2e-and-diagnosis.yml").read_text(
        encoding="utf-8"
    )
    assert "scripts/run_plm_e2e_and_diag.py" in wf
    assert "TACOSM-PLM-AND-DIAG-001" in wf
    assert "workflow_dispatch:" in wf
