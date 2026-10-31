"""LANGCAP104 constructed contract checks only. No optimizer and no real fit."""
from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path
import hashlib
import json
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/"research/langcap104/train.py"
SPEC=spec_from_file_location("langcap104_test_worker",PATH)
worker=module_from_spec(SPEC)
SPEC.loader.exec_module(worker)

def fake_frozen(tmp_path):
    root=tmp_path/"langcap104"
    root.mkdir()
    target=root/"FAIR_COMPARISON.json"
    payload={"study_id":"LANGCAP104_FROZEN_TRY", "study_type":"single_model_training",
       "training_enabled":True,"registration_status":"FROZEN_APPROVED_AUTHENTIC_FIT",
       "real_world_source":{"sha256":worker.EXPECTED_SOURCE_SHA},
       "no_architecture_superiority_claim":True,
       "compute":{"approved_gpu_minutes":20,"training_steps":2560,
          "t4_throughput_verified":{"status":"MEASURED"},
          "full_training_flops_verified":{"status":"MEASURED"}}}
    target.write_text(json.dumps(payload))
    return target

def test_missing_frozen_registration_blocks_before_torch_and_data(tmp_path):
    path=tmp_path/"langcap104"/"FAIR_COMPARISON.json"
    with pytest.raises(worker.NotAuthorized,match="missing"):
        worker.verify_authorization(path)
    assert "torch" not in worker.__dict__

def test_source_only_draft_cannot_authorize_fit(tmp_path):
    path=fake_frozen(tmp_path)
    d=json.loads(path.read_text())
    d["registration_status"]="REVIEW_DRAFT_NOT_A_COMPLETE_AUTHORIZED_TRAINING_CONTRACT"
    path.write_text(json.dumps(d))
    with pytest.raises(worker.NotAuthorized,match="approved frozen"):
        worker.verify_authorization(path)

def test_gpu_and_signed_credentials_are_never_default(tmp_path,monkeypatch):
    path=fake_frozen(tmp_path)
    monkeypatch.delenv("LANGCAP104_GPU_EXECUTION_ENABLED",raising=False)
    with pytest.raises(worker.NotAuthorized,match="protected GPU"):
        worker.verify_authorization(path)
    monkeypatch.setenv("LANGCAP104_GPU_EXECUTION_ENABLED","true")
    monkeypatch.delenv("LANGCAP104_FROZEN_MANIFEST_SHA256",raising=False)
    with pytest.raises(worker.NotAuthorized,match="manifest hash"):
        worker.verify_authorization(path)
    monkeypatch.setenv("LANGCAP104_FROZEN_MANIFEST_SHA256",
                       hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.delenv("TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX",raising=False)
    with pytest.raises(worker.NotAuthorized,match="verification key"):
        worker.verify_authorization(path)

def test_frozen_source_compute_and_checkpoint_requirements_fail_closed(tmp_path,monkeypatch):
    path=fake_frozen(tmp_path)
    monkeypatch.delenv("LANGCAP104_GPU_EXECUTION_ENABLED",raising=False)
    base=json.loads(path.read_text())
    for field in ("approved_gpu_minutes","training_steps",
                  "t4_throughput_verified","full_training_flops_verified"):
        trial=json.loads(json.dumps(base));trial["compute"].pop(field)
        path.write_text(json.dumps(trial))
        with pytest.raises(worker.NotAuthorized,match="Approved GPU quota"):
            worker.verify_authorization(path)
    trial=json.loads(json.dumps(base))
    trial["real_world_source"]["sha256"]="0"*64
    path.write_text(json.dumps(trial))
    with pytest.raises(worker.NotAuthorized,match="CPython source"):
        worker.verify_authorization(path)
