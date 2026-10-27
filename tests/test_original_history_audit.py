"""Constructed permission/trace falsifiers, never original-source performance."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tac_osm.history_audit_authorization import (
    CHECKS, AuditBlocked, audit_binding, canonical, verify_authorizations,
)
from scripts.audit_original_history import preflight
from scripts.original_history_trace import paired_gain_ci, verify_causal_features, verify_implementation

NOW = datetime(2026, 10, 27, tzinfo=timezone.utc)


def permissions(binding):
    # Test keys are ephemeral fixtures; these never authorize a real source run.
    keys, receipts, signers = {}, {}, {}
    for role in CHECKS:
        signer = Ed25519PrivateKey.generate()
        signers[role] = signer
        keys[role] = signer.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        payload = {"schema": "tac-osm.history-audit-authorization.v1", "role": role,
            "binding": binding, "checks": {k: "PASS" for k in CHECKS[role]},
            "expires_at": "2026-10-28T00:00:00Z"}
        receipts[role] = {"payload": payload, "signature_hex": signer.sign(canonical(payload)).hex()}
    return receipts, keys, signers


def test_two_external_role_signatures_bind_entire_plan_code_and_models():
    plan = {"study_id": "FIXTURE", "implementation_commit": "a" * 40,
            "original_source_sha256": "b" * 64,
            "models": [{"arm": "fixture", "seed": 1, "sha256": "c" * 64}]}
    binding = audit_binding(plan, {"fixture": True}, {"runner.py": "d" * 64})
    receipts, keys, _ = permissions(binding)
    assert verify_authorizations(binding, receipts, keys, now=NOW)["scientific_qualification"] is False
    for key in ("plan_sha256", "proposal_sha256", "execution_files_sha256", "source_sha256"):
        changed = binding | {key: "e" * 64}
        with pytest.raises(AuditBlocked):
            verify_authorizations(changed, receipts, keys, now=NOW)
    changed = deepcopy(binding)
    changed["checkpoints"][0]["sha256"] = "e" * 64
    with pytest.raises(AuditBlocked):
        verify_authorizations(changed, receipts, keys, now=NOW)


@pytest.mark.parametrize("fault", ["missing_source", "missing_evaluator", "same_key", "expired",
                                  "failed_check", "wrong_role", "spoof_signature", "extra_field"])
def test_independent_authorization_fails_closed(fault):
    binding = {"fixture_only": True}
    receipts, keys, signers = permissions(binding)
    if fault.startswith("missing_"):
        receipts[fault.removeprefix("missing_")] = None
    elif fault == "same_key":
        keys["evaluator"] = keys["source"]
    elif fault == "spoof_signature":
        receipts["evaluator"]["signature_hex"] = "0" * 128
    else:
        payload = receipts["evaluator"]["payload"]
        if fault == "expired":
            payload["expires_at"] = "2026-10-26T00:00:00Z"
        elif fault == "failed_check":
            payload["checks"]["independent_evaluator_and_ACLs"] = "INCONCLUSIVE"
        elif fault == "wrong_role":
            payload["role"] = "source"
        else:
            payload["force"] = True
        receipts["evaluator"]["signature_hex"] = signers["evaluator"].sign(canonical(payload)).hex()
    with pytest.raises(AuditBlocked):
        verify_authorizations(binding, receipts, keys, now=NOW)


def test_missing_authorization_blocks_before_any_source_or_model_import(tmp_path, monkeypatch):
    for name in ("TAC_OSM_SOURCE_VERIFIER_PUBLIC_KEY_HEX", "TAC_OSM_EVALUATOR_VERIFIER_PUBLIC_KEY_HEX"):
        monkeypatch.delenv(name, raising=False)
    _, result = preflight()
    assert result["status"] == "BLOCKED_INDEPENDENT_AUTHORIZATION"
    assert result["compute_decision"]["backend"] == "github_actions"
    output = tmp_path / "blocker.json"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/audit_original_history.py"),
        "--execute", "--implementation-root", str(tmp_path / "NO_MODEL"),
        "--source", str(tmp_path / "NO_SOURCE"), "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    r = json.loads(output.read_text())
    assert r["evaluation_executed"] is False
    assert r["status"] == "BLOCKED_INDEPENDENT_AUTHORIZATION"
    assert "original_history_trace" not in r.get("reason", "")
    assert "missing" in r["reason"]
    # A second attempt must preserve the original negative receipt.
    again = subprocess.run([sys.executable, str(ROOT / "scripts/audit_original_history.py"),
        "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
    assert again.returncode == 2
    assert json.loads(output.read_text()) == r


def test_pinned_source_rejects_modified_or_untracked_python(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args]).decode().strip()
    git("init", "-q")
    (root / "src").mkdir()
    file = root / "src/original.py"
    file.write_text("# constructed fixture\n")
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
    commit = git("rev-parse", "HEAD")
    assert verify_implementation(root, commit) == root
    file.write_text("raise RuntimeError('changed')\n")
    with pytest.raises(AuditBlocked, match="modified"):
        verify_implementation(root, commit)
    file.write_text("# constructed fixture\n")
    (root / "src/injected.py").write_text("# unexpected module\n")
    with pytest.raises(AuditBlocked, match="extra"):
        verify_implementation(root, commit)


def test_station_bootstrap_preserves_source_groups_and_ties():
    assert paired_gain_ci([0, 0, 1, 1, 2, 2], [1.] * 6, [1.] * 6, draws=100, seed=1) == [0., 0.]
    assert paired_gain_ci([0, 0, 1, 1, 2, 2], [1.] * 6, [0.] * 6, draws=100, seed=1) == [1., 1.]
    with pytest.raises(AuditBlocked):
        paired_gain_ci([0, 0], [1., 1.], [0., 0.], draws=10, seed=1)


def test_prefix_reconstruction_rejects_future_fitted_transform_and_bad_cache():
    import torch
    from types import SimpleNamespace
    raw = torch.arange(12 * 10 * 13, dtype=torch.float32).reshape(12, 10, 13) / 1000 + 1
    raw[0, 2, 0] = float("nan")
    x = raw.clone()
    x[:, :, :6] = torch.log1p(x[:, :, :6])
    past = x[:, :4]
    seen = torch.isfinite(past)
    n = seen.sum((0, 1)).clamp_min(1)
    center = torch.where(seen, past, 0).sum((0, 1)) / n
    spread = (torch.where(seen, past - center, 0).square().sum((0, 1)) / n).clamp_min(1e-6).sqrt()
    normalized = (x - center) / spread
    valid = torch.isfinite(normalized)
    values = torch.where(valid, normalized, 0)
    last, age_cache = torch.zeros_like(values), torch.zeros_like(values)
    recent = torch.zeros((12, 13))
    age = torch.full_like(recent, 256.)
    for t in range(10):
        last[:, t] = recent
        age = torch.where(valid[:, t], 0., (age + 1).clamp(max=256))
        age_cache[:, t] = age
        recent = torch.where(valid[:, t], values[:, t], recent)
    d = dict(raw=raw, center=center, spread=spread, valid=valid, values=values, last=last, age=age_cache)
    source = SimpleNamespace(hour=lambda _: 4)
    verify_causal_features(d, source, 8)
    with pytest.raises(AssertionError):
        verify_causal_features(d | {"center": center + 1}, source, 8)
    last[0, 1, 0] = 9999
    with pytest.raises(AuditBlocked, match="last-value cache"):
        verify_causal_features(d, source, 8)


def test_actual_original_graph_trace_with_constructed_inputs_only():
    original = os.environ.get("TAC_OSM_ORIGINAL_HISTORY_SOURCE")
    if not original:
        pytest.skip("pinned original-code checkout required; dedicated history-audit CI supplies it")
    # Isolate original model imports from the full suite's current package graph.
    code = r'''
import sys
from pathlib import Path
sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "src")]
from scripts.original_history_trace import verify_implementation, load_original_modules, trace_case
from tac_osm.historical_read_controls import select_intervening_writes
import torch
root = verify_implementation(Path(sys.argv[1]), "0d9c9fbc8c5ed94762c9da129379158d36125e7f")
source, fit = load_original_modules(root)
torch.set_num_threads(1)
torch.manual_seed(101)
m = fit.create_model(101, "assoc_historic_query").eval()
values = torch.randn(12, 100, 13)
d = dict(values=values, valid=torch.ones_like(values, dtype=torch.bool),
         age=torch.zeros_like(values), last=torch.zeros_like(values), raw=object())
with torch.no_grad():
    result = trace_case(m, d, source, site=0, anchor=24, lag=64, control_count=4, control_seed=14014)
    assert result["memory_update_max_abs_error"] <= 2e-6
    assert result["prediction_equivalence_max_abs_error"] <= 2e-6
    assert len(result["trace"]) == 88
    selected = result["control_selection"]["selected_hours"]
    assert all(24 < t < 88 for t in selected)
    actual = {r["hour"]: r["key_sha256"] for r in result["trace"] if r["directional_write_nonzero"]}
    assert all(c["key_sha256"] == actual[c["hour"]] for c in result["controls_PRIVILEGED"])
    # Values, masks and caches strictly AFTER query must not change output;
    # opaque raw means a label/raw lookup anywhere inside trace_case fails.
    poisoned = {k: v.clone() if torch.is_tensor(v) else v for k, v in d.items()}
    for k in ("values", "age", "last"):
        poisoned[k][:, 89:] = 1e9
    poisoned["valid"][:, 89:] = False
    rerun = trace_case(m, poisoned, source, site=0, anchor=24, lag=64, control_count=4, control_seed=14014)
    assert rerun == result
    d["valid"][0, 24] = False
    try:
        trace_case(m, d, source, site=0, anchor=24, lag=64, control_count=4, control_seed=14014)
    except ValueError as exc:
        assert "target missing" in str(exc)
    else:
        raise AssertionError("nonwrite source target accepted")
print("PASS_ORIGINAL_GRAPH_FIXTURE_ONLY_NO_REAL_SOURCE")
'''
    result = subprocess.run([sys.executable, "-c", code, original], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS_ORIGINAL_GRAPH_FIXTURE_ONLY" in result.stdout
