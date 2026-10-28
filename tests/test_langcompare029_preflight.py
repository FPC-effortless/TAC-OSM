"""LANGCOMPARE029 no-fit safety checks, not trained benchmark evidence."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts/langcompare029_preflight.py"
spec=importlib.util.spec_from_file_location("langcompare029_preflight_test",SCRIPT)
p=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

def test_parameter_budget_formula_and_spread():
    assert p.BDH_ANALYTICAL==501504
    assert p.MAMBA_ANALYTICAL==501884
    assert p.TRANSFORMER==496960
    assert (max(p.EXPECTED)-min(p.EXPECTED))/max(p.EXPECTED)<0.01

def test_transformer_actual_and_no_future_leakage():
    model=p.ByteTransformer()
    assert p.parameter_count(model)=={"trainable":496960,"deployed":496960}
    assert p.compare_prefix(model,"cpu",1e-5)<1e-5

def test_exact_existing_native_graph_not_mutated_and_causal():
    m=p.load_tac()
    assert p.parameter_count(m)=={"trainable":500122,"deployed":500122}
    assert p.compare_prefix(m,"cpu",1e-5)<1e-5

def test_preflight_never_claims_full_four_arm_or_training_without_mamba():
    r=p.preflight(bdh_path=None,attempt_mamba=False,check_causality=False)
    assert r["status"]=="NO_FIT_PARTIAL_PREFLIGHT_NOT_READY_TO_TRAIN"
    assert r["optimizer_steps"]==0 and r["gpu_minutes"]==0
    assert r["DEV_NLL"]=="NOT_RUN" and not r["training_authorized"]
    assert r["source"]["fit_docs"]==14 and r["source"]["exposed_dev_docs"]==4

def test_complete_gate_fails_closed():
    with pytest.raises(RuntimeError,match="INCOMPLETE"):
        p.preflight(bdh_path=None,attempt_mamba=False,
                    require_complete=True,check_causality=False)
