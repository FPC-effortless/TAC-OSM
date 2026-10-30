"""No-fit constructed checks for 10M parity controls and corpus source boundary."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def test_three_control_counts_and_trained_status():
    c = module("lang10m_control_counts", "controls.py")
    assert set(c.CONTROL_COUNTS) == {"gru", "selective_diagonal_ssm", "causal_transformer"}
    for name, expected in c.CONTROL_COUNTS.items():
        model = c.make_control(name)
        assert sum(p.numel() for p in model.parameters()) == expected
        assert abs(expected / c.TARGET_TAC - 1) < .01
    assert c.status()["trained"] is False
    assert c.status()["training_compute_parity"] == "UNMEASURED"


@pytest.mark.parametrize("arm", ["gru", "selective_diagonal_ssm", "causal_transformer"])
def test_causal_future_token_poison(arm):
    c = module("lang10m_control_poison", "controls.py")
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    model = c.make_control(arm).eval()
    x = torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long)
    y = x.clone()
    y[:, -1] = 4095
    with torch.no_grad():
        before = model(x)
        after = model(y)
    assert before.shape == (1, 5, 4096)
    torch.testing.assert_close(before[:, :-1], after[:, :-1], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("arm", ["gru", "selective_diagonal_ssm", "causal_transformer"])
def test_real_autograd_parameter_connections_on_constructed_tokens(arm):
    c = module("lang10m_control_grad", "controls.py")
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    model = c.make_control(arm).train()
    x = torch.tensor([[44, 30, 12, 200]], dtype=torch.long)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits[:, -1], torch.tensor([31], dtype=torch.long)
    )
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, (arm, name)
        assert torch.isfinite(param.grad).all(), (arm, name)
    assert torch.isfinite(loss)


def fixture_manifest(tmp_path):
    gate = module("lang10m_gate_test", "corpus_gate.py")
    notice = tmp_path / "LICENSE"
    notice.write_text("Illustrative license fixture only; not real-source authorization.\n")
    entries = []
    for idx, (name, kind, partition) in enumerate([
        ("fitcode", "code", "FIT"),
        ("fitprose", "natural_language", "FIT"),
        ("dev", "code", "SELECT_DEV"),
        ("transfer", "code", "TRANSFER_DEV"),
    ]):
        path = tmp_path / (name + ".txt")
        path.write_text(f"Distinct example {idx}. Content to exercise file hash checking.\n")
        raw = path.read_bytes()
        entries.append({
            "partition": partition, "kind": kind, "publisher_group": name,
            "repo": "fictional/"+name, "original_path": name+".txt",
            "source_commit": f"{idx+1:040x}", "sha256": gate.sha256(raw),
            "git_blob_sha1": gate.git_blob(raw), "local_path": path.name,
            "license_path": notice.name, "license_id": "FIXTURE",
            "license_sha256": gate.sha256(notice.read_bytes()),
        })
    doc = {"study": "LANG10M002", "schema": "original-source-v1", "sources": entries}
    mf = tmp_path / "sources.json"
    mf.write_text(json.dumps(doc))
    return gate, mf, doc


def test_manifest_validation_no_fit_and_no_unsupported_promotion(tmp_path):
    gate, mf, _ = fixture_manifest(tmp_path)
    result = gate.validate_manifest(mf)
    assert result["FIT_sources"] == 2
    assert result["FIT_token_instances"] is None
    assert result["FIT_token_instances_minimum_50M_met"] is False
    assert result["claims"]["model_trained"] is False
    assert result["near_duplicate_and_semantic_contamination"].startswith("INCONCLUSIVE")


def test_manifest_rejects_cross_publisher_partition_and_source_corruption(tmp_path):
    gate, mf, doc = fixture_manifest(tmp_path)
    doc["sources"][2]["publisher_group"] = "fitcode"
    mf.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="publisher group"):
        gate.validate_manifest(mf)
    doc["sources"][2]["publisher_group"] = "dev"
    mf.write_text(json.dumps(doc))
    (tmp_path/"fitcode.txt").write_text("tampered after manifest freeze")
    with pytest.raises(ValueError, match="source digest"):
        gate.validate_manifest(mf)


def test_manifest_rejects_test_sources_and_duplicate_files(tmp_path):
    gate, mf, doc = fixture_manifest(tmp_path)
    doc["sources"][3]["partition"] = "TEST"
    mf.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="partition"):
        gate.validate_manifest(mf)
    doc["sources"][3]["partition"] = "TRANSFER_DEV"
    doc["sources"][3]["local_path"] = doc["sources"][0]["local_path"]
    mf.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="same local file"):
        gate.validate_manifest(mf)
