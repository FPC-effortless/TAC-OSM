"""No-fit engineering checks; no claim of language capability or PNDS proof."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent


def local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_parameter_count_and_memory_lineage():
    m = local_module("lang10m_model_count", "model.py")
    model = m.TACLanguage10M001()
    assert m.EXPECTED_PARAMETERS == 10_001_188
    assert sum(p.numel() for p in model.parameters()) == 10_001_188
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 10_001_188
    assert len(model.layernorm) == 5  # first-stage LN, four new causal blocks
    rec = model.architecture_record()
    assert rec["native_memory"] == "8x8"
    assert rec["base_concept_slots"] == "6x16"
    assert rec["upper_blocks"] == 4
    assert rec["window_reset_only"] and not rec["trained"]
    assert rec["composed_pnds_certification"] == "OPEN"


def test_causal_prefix_no_future_leakage():
    m = local_module("lang10m_model_causal", "model.py")
    torch.manual_seed(1010)
    model = m.TACLanguage10M001().eval()
    x = torch.tensor([[5, 10, 20, 30, 40]], dtype=torch.long)
    y = x.clone()
    y[0, -1] = 999
    with torch.no_grad():
        a = model(x)
        b = model(y)
    assert a.shape == (1, 5, 4096)
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a[:, :-1, :], b[:, :-1, :], atol=2e-4, rtol=2e-5)


def test_actual_backward_to_native_and_upper_layers():
    m = local_module("lang10m_model_backward", "model.py")
    torch.manual_seed(1010)
    model = m.TACLanguage10M001().train()
    logits = model(torch.tensor([[11, 12, 13, 14]], dtype=torch.long))
    loss = torch.nn.functional.cross_entropy(
        logits[:, -1, :], torch.tensor([17], dtype=torch.long)
    )
    loss.backward()
    for name, tensor in [
        ("native_key", model.native_key.weight),
        ("wave_decay", model.wave_decay.weight),
        ("concept_value", model.concept_value.weight),
        ("upper_recurrence", model.layernorm[1].recurrence.input_proj.weight),
        ("upper_ffn", model.layernorm[1].up.weight),
        ("output", model.ffn_out.weight),
    ]:
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name
    # Gradient existence is a structural check; nonzero utility is not implied.


def test_fit_only_manifest_fails_closed(tmp_path):
    t = local_module("lang10m_tokenizer_manifest", "tokenizer_fit.py")
    raw = tmp_path / "training.py"
    raw.write_text("def add(a, b):\n    return a + b\n")
    row = {
        "path": str(raw),
        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "license": "MIT",
        "publisher": "verified/example",
        "commit": "deadbeef",
        "partition": "FIT",
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"partition": "FIT", "sources": [row]}))
    assert t.validate_fit_manifest(manifest) == [raw.resolve()]
    wrong = {"partition": "DEV", "sources": [dict(row, partition="DEV")]}
    manifest.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="FIT"):
        t.validate_fit_manifest(manifest)
    row["sha256"] = "0" * 64
    manifest.write_text(json.dumps({"partition": "FIT", "sources": [row]}))
    with pytest.raises(ValueError, match="SHA256"):
        t.validate_fit_manifest(manifest)
