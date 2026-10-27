"""Verify external permission for frozen original-source diagnostics only.

This is not the training/reward-promotion gate. No signing capability is
provided. Keys must be independently provisioned outside the actor worktree;
signatures cannot establish actual OS/identity separation on their own.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json


class AuditBlocked(ValueError):
    pass


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


CHECKS = {
    "source": {"publisher_and_license", "original_source_identity",
               "authorized_FIT_scope", "protected_TEST_sealed"},
    "evaluator": {"independent_evaluator_and_ACLs", "frozen_evaluation_policy",
                  "frozen_checkpoint_replay_authorized", "no_training_or_promotion"},
}


def audit_binding(plan, proposal, execution_files):
    return {
        "study_id": plan["study_id"], "plan_sha256": digest(plan),
        "proposal_sha256": digest(proposal),
        "execution_files_sha256": digest(execution_files),
        "implementation_commit": plan["implementation_commit"],
        "source_sha256": plan["original_source_sha256"],
        "checkpoints": [{k: row[k] for k in ("arm", "seed", "sha256")}
                        for row in plan["models"]],
        "scope": "frozen_FIT_historical_memory_diagnostic_only",
        "training_permitted": False, "TEST_permitted": False,
    }


def verify_authorizations(binding, receipts, public_keys, *, now=None):
    """Require two differently keyed, role-specific, unexpired authorizations.

    Provisioning and independent source investigation are operator/evaluator
    responsibilities. Test-local signers never authorize a real run.
    """
    if set(receipts) != set(CHECKS) or set(public_keys) != set(CHECKS):
        raise AuditBlocked("independent source AND evaluator authorizations required")
    if any(not public_keys[role] or receipts[role] is None for role in CHECKS):
        raise AuditBlocked("independent source/evaluator receipts or trusted public keys missing")
    try:
        raw_keys = {role: bytes.fromhex(value) for role, value in public_keys.items()}
    except (TypeError, ValueError) as exc:
        raise AuditBlocked("invalid independently pinned public key") from exc
    if any(len(key) != 32 for key in raw_keys.values()) or len(set(raw_keys.values())) != 2:
        raise AuditBlocked("distinct independently pinned source and evaluator keys required")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    now = now or datetime.now(timezone.utc)
    verified = {}
    for role in CHECKS:
        envelope = receipts[role]
        try:
            if set(envelope) != {"payload", "signature_hex"}:
                raise ValueError("invalid envelope")
            payload = envelope["payload"]
            Ed25519PublicKey.from_public_bytes(raw_keys[role]).verify(
                bytes.fromhex(envelope["signature_hex"]), canonical(payload))
            if (set(payload) != {"schema", "role", "binding", "checks", "expires_at"}
                    or payload["schema"] != "tac-osm.history-audit-authorization.v1"
                    or payload["role"] != role or payload["binding"] != binding
                    or set(payload["checks"]) != CHECKS[role]
                    or any(v != "PASS" for v in payload["checks"].values())):
                raise ValueError("scope, role, binding or independent checks mismatch")
            expiry = datetime.fromisoformat(payload["expires_at"].replace("Z", "+00:00"))
            if expiry.tzinfo is None or expiry <= now:
                raise ValueError("expired or timezone-ambiguous authorization")
        except Exception as exc:
            raise AuditBlocked(f"{role} authorization invalid, expired or untrusted") from exc
        verified[role] = {"receipt_sha256": digest(envelope),
                          "public_key_sha256": hashlib.sha256(raw_keys[role]).hexdigest()}
    return {"status": "AUTHORIZED_FROZEN_DIAGNOSTIC_ONLY", "verified": verified,
            "scientific_qualification": False}
