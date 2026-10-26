"""Constructed cryptographic falsifiers for all prospective variant runners.

Signatures in these tests are generated in process and are NOT independent
real-source verification or permission isolation. Production private key
must never enter model training/repository agents.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from tac_osm.research_reward_runner import (
    FAMILIES, SCHEMA, RewardEvidenceError, binding, verify_reward_attestation
)
from scripts.validate_research_parity import check_manifest, ParityError

ROOT = Path(__file__).resolve().parents[1]


def make_manifest():
    return {
        "study_id": "ME999_CONSTRUCTED_ONLY",
        "study_type": "single_model_training",
        "schema_version": "tac-osm.parity.v1",
        "registered_before_first_fit": True,
        "real_world_source": {"sha256": "a" * 64, "observed_real_outcomes": True},
        "models": [
            {"variant": family, "seed": 101, "trainable_parameters": 100000,
             "deployed_parameters": 100000,
             "measured": {"trainable_parameters": 100000,
                          "deployed_parameters": 100000,
                          "checkpoint_sha256": f"{index+1:x}" * 64}}
            for index, family in enumerate(FAMILIES)
        ],
    }


def receipt(manifest, phase="final", *, signer=None):
    signer = signer or Ed25519PrivateKey.generate()
    checks = (("protected_assets", "source_provenance",
               "goal_evaluator_separation", "actor_access_isolation") if phase == "preflight"
              else ("protected_assets", "reward_recomputation", "goal_recomputation",
                    "reward_noninterference", "proxy_goal_alignment"))
    payload = {
        "schema": SCHEMA, "phase": phase, "binding": binding(manifest),
        "source_authority": "independently_verified_real_source",
        "evaluator_executed_outside_actor": True,
        "reward_scorer_sha256": "b" * 64,
        "goal_evaluator_sha256": "c" * 64,
        "access_policy_sha256": "d" * 64,
        "protected_assets_sha256": "e" * 64,
        "evaluation_policy_sha256": "f" * 64,
        "checks": {name: "PASS" for name in checks},
    }
    if phase == "final":
        payload["evaluated"] = [
            {"variant": row["variant"], "seed": row["seed"],
             "checkpoint_sha256": row["measured"]["checkpoint_sha256"],
             "outcome_evidence_sha256": "3" * 64, "observation_count": 4}
            for row in sorted(manifest["models"], key=lambda x: (x["variant"], x["seed"]))
        ]
    def sign(payload):
        return {"payload": payload, "signature_hex": signer.sign(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hex()}
    public = signer.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw).hex()
    return sign(payload), public, sign


def test_complete_six_variant_prefit_and_final_binding():
    m = make_manifest()
    for phase in ("preflight", "final"):
        evidence, public, _ = receipt(m, phase)
        result = verify_reward_attestation(
            m, phase, evidence, trusted_public_key_hex=public)
        assert result["status"] == "PASS_SIGNED_EXTERNAL_AUDIT"
        assert result["variant_seed_count"] == 6


@pytest.mark.parametrize("field,wrong", [
    ("source_authority", "self_reported"),
    ("evaluator_executed_outside_actor", False),
    ("reward_scorer_sha256", "c" * 64),
    ("access_policy_sha256", ""),
])
def test_signed_but_untrusted_or_incomplete_claims_fail(field, wrong):
    m = make_manifest()
    a, key, sign = receipt(m)
    payload = deepcopy(a["payload"])
    payload[field] = wrong
    with pytest.raises(RewardEvidenceError):
        verify_reward_attestation(m, "final", sign(payload), trusted_public_key_hex=key)


@pytest.mark.parametrize("missing", [
    "reward_recomputation", "goal_recomputation", "reward_noninterference",
    "proxy_goal_alignment", "protected_assets",
])
def test_no_independent_final_check_can_be_absent_or_inconclusive(missing):
    m = make_manifest()
    a, key, sign = receipt(m)
    changed = deepcopy(a["payload"])
    del changed["checks"][missing]
    with pytest.raises(RewardEvidenceError, match="mandatory"):
        verify_reward_attestation(m, "final", sign(changed), trusted_public_key_hex=key)
    changed = deepcopy(a["payload"])
    changed["checks"][missing] = "INCONCLUSIVE"
    with pytest.raises(RewardEvidenceError, match="FAIL or INCONCLUSIVE"):
        verify_reward_attestation(m, "final", sign(changed), trusted_public_key_hex=key)


def test_untrusted_signer_fails_even_when_every_claim_says_pass():
    m = make_manifest()
    envelope, _, _ = receipt(m)
    _, wrong_key, _ = receipt(m)
    with pytest.raises(RewardEvidenceError, match="signature"):
        verify_reward_attestation(m, "final", envelope,
                                  trusted_public_key_hex=wrong_key)


def test_changed_checkpoint_invalidates_signed_final_attestation():
    m = make_manifest()
    envelope, key, _ = receipt(m)
    m["models"][0]["measured"]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(RewardEvidenceError, match="checkpoint"):
        verify_reward_attestation(m, "final", envelope,
                                  trusted_public_key_hex=key)


def test_missing_one_variant_checkpoint_blocks_six_variant_promotion():
    m = make_manifest()
    envelope, key, sign = receipt(m)
    corrupt = deepcopy(envelope["payload"])
    corrupt["evaluated"].pop()
    with pytest.raises(RewardEvidenceError, match="per-model"):
        verify_reward_attestation(m, "final", sign(corrupt),
                                  trusted_public_key_hex=key)


def test_source_and_variant_rebinding_rejected():
    m = make_manifest()
    a, key, _ = receipt(m, "preflight")
    modified = deepcopy(m)
    modified["real_world_source"]["sha256"] = "f" * 64
    with pytest.raises(RewardEvidenceError, match="different"):
        verify_reward_attestation(modified, "preflight", a,
                                  trusted_public_key_hex=key)
    modified = deepcopy(m)
    modified["models"][0]["variant"] = "new variant"
    with pytest.raises(RewardEvidenceError, match="different"):
        verify_reward_attestation(modified, "preflight", a,
                                  trusted_public_key_hex=key)


def test_original_resource_parity_cannot_pass_new_study_without_certificate():
    m = make_manifest()
    with pytest.raises(ParityError, match="independent reward/outcome/access"):
        check_manifest(m, "preflight")
    with pytest.raises(ParityError, match="independent reward/outcome/access"):
        check_manifest(m, "final")


def test_parity_accepts_independent_certificate_for_all_registered_variants():
    m = make_manifest()
    for phase in ("preflight", "final"):
        envelope, pub, _ = receipt(m, phase)
        result = check_manifest(m, phase, reward_attestation=envelope,
                                trusted_reward_public_key_hex=pub)
        assert result["reward_integrity"]["variant_seed_count"] == 6
        assert not result["eligible_for_architecture_superiority"]


def test_cli_without_operator_key_fails_even_with_signed_receipt(tmp_path):
    m = make_manifest()
    env, key, _ = receipt(m, "preflight")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(m))
    cert = tmp_path / "certificate.json"
    cert.write_text(json.dumps(env))
    import os
    child_env = dict(os.environ)
    child_env.pop("TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX", None)
    res = subprocess.run([sys.executable, str(ROOT / "scripts/validate_research_parity.py"),
                          "--manifest", str(manifest_path), "--phase", "preflight",
                          "--reward-evidence", str(cert)],
                         cwd=ROOT, env=child_env, capture_output=True, text=True)
    assert res.returncode == 2
    assert "independent reward/outcome/access" in res.stderr


def test_cli_prefit_operator_key_verifies_signed_fixture(tmp_path):
    m = make_manifest()
    env, key, _ = receipt(m, "preflight")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(m))
    cert_path = tmp_path / "certificate.json"
    cert_path.write_text(json.dumps(env))
    import os
    child_env = dict(os.environ, TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX=key)
    res = subprocess.run([sys.executable, str(ROOT / "scripts/validate_research_parity.py"),
                          "--manifest", str(manifest_path), "--phase", "preflight",
                          "--reward-evidence", str(cert_path)],
                         cwd=ROOT, env=child_env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout)["reward_integrity"]["variant_seed_count"] == 6
