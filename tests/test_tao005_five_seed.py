"""TAOFINAL005: preregistered original-source five-seed research gates."""
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"src",ROOT/"research/mathematical-engineering/langmem086"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from scripts import tao005_five_seed as study


def test_tao005_manifest_has_exact_same_param_count_25_original_models():
    m=study.registry()
    assert m["study_type"]=="single_model_training"
    assert m["execution_level"]=="internal_research"
    assert m["no_architecture_superiority_claim"] is True
    assert len(m["models"])==25
    assert {r["seed"] for r in m["models"]}==set(study.SEEDS)
    assert {r["name"] for r in m["models"]}=={
        f"{a}_seed{s}" for a in study.ARMS for s in study.SEEDS}
    assert all(r["trainable_parameters"]==100521 and r["deployed_parameters"]==100521 for r in m["models"])
    assert m["budget"]["updates_per_arm_seed"]==64
    assert m["budget"]["real_targets_per_arm_seed"]==65536
    assert set(m["report_dev_owners"])=={
        "Doc/tutorial/introduction.rst","Doc/tutorial/venv.rst"}


def test_tao005_original_frozen_real_seed_schedules():
    plans=[]
    for seed in study.SEEDS:
        docs,batches,schedule=study.frozen_real_fits(seed)
        assert len(docs)==14 and len(batches)==64
        assert schedule["seed"]==seed and schedule["targets"]==65536
        assert len(schedule["sample_bytes_sha256"])==64
        plans.append(schedule["sample_order_sha256"])
    assert len(set(plans))==5


def test_tao005_refuses_unselected_lr_and_early_report_access(tmp_path,monkeypatch):
    monkeypatch.setattr(study,"TUNING",tmp_path/"missing_tuning_results.json")
    with pytest.raises(FileNotFoundError):
        study.selection()
    # No report source is accessed when the 25 original models are missing.
    with pytest.raises(FileNotFoundError):
        study.report(tmp_path)
