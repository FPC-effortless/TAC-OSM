"""Fresh owner-scoped exception gate; the old one-off approval cannot transfer."""
import hashlib
import json
from pathlib import Path

SOURCE_SHA256 = "fc3c2bfb88f977f3ec0bb8226c13a4ef710f95ef23a1eea5677abe4777b5b40f"
SCIENTIFIC_SHA256 = "29083ee944b0f945ca605c37f9daae935a3b938198ba07c2df2d656460dd69d3"
SCOPE = {'families': ['context_legacy', 'context_repaired'], 'seed': 18001, 'updates_per_arm': 128, 'optimizer_updates_total': 256, 'optimizer_regime': 'default_epsilon1e-8', 'CPU_seconds_max': 360, 'wall_seconds_max': 600, 'retry_authorized': False, 'model_variants': {'context_legacy': 'COLLECT018-CONTEXT_FIRST', 'context_repaired': 'FLOCK013-CONTEXT-ADJOINT'}}
SCIENCE_FIELDS = ("families", "seed", "parameter_counts", "regimes", "stages", "optimizer_other_settings", "objective", "fixture_files", "observed_coordinates", "model_variants")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise PermissionError(reason)


def inspect(root, protocol, grant):
    root = Path(root)
    try:
        require(grant.get("scope_id") == "FLOCK013_CONTEXT_PAIRED_V2_20261103" and grant.get("protocol") == "LOCAL_BINDING_LEARNABILITY_V1", "Wrong repair release identity")
        require(hashlib.sha256(canonical({k: protocol[k] for k in SCIENCE_FIELDS})).hexdigest() == SCIENTIFIC_SHA256, "Frozen repair scientific scope changed")
        require(digest(root/"SOURCE_MANIFEST.json") == SOURCE_SHA256 == grant.get("source_manifest_sha256"), "Frozen source identity changed")
        require(digest(root/"PROTOCOL.json") == grant.get("protocol_sha256"), "Protocol grant differs")
        require(digest(root/"CODE_MANIFEST.json") == grant.get("code_manifest_sha256"), "Code grant differs")
        for name, row in json.loads((root/"CODE_MANIFEST.json").read_bytes())["files"].items():
            require(digest(root/name) == row["sha256"], "Changed repair runner source")
        for name, row in protocol["admission"]["files"].items():
            require(digest(root/name) == row["sha256"], "Changed approval/initializer metadata")
        require(grant.get("stage") == "fixed" and grant.get("regime") == "default" and grant.get("retry_authorized") is False, "Unreleased stage, regime or retry")
        require(grant.get("caps") == protocol["stages"]["fixed"]["caps"], "Resource release differs")
        approval = json.loads((root/"OWNER_AUTHORIZATION.json").read_bytes())
        require(approval.get("schema") == "flock013.owner-synthetic-repair-exception.v2" and approval.get("scope") == SCOPE, "Old or different owner exception does not transfer")
        require(approval.get("approved") is True, "A fresh owner synthetic-fit exception has not been approved")
        require(approval.get("previous_one_off_exception_transfers") is False, "Previous one-off authority cannot transfer")
        require(approval.get("basis") == "DIRECT_USER_RESPONSE_TO_THIS_EXACT_REPAIR_TRIAL" and str(approval.get("user_response", "")).strip().lower() in {"yes", "proceed", "approved"}, "Missing direct scope-specific approval record")
        require(grant.get("owner_authorization_sha256") == digest(root/"OWNER_AUTHORIZATION.json"), "Approval grant identity differs")
        require(protocol.get("training_authorized") is True and grant.get("authorized") is True, "Training release is disabled")
        return {"status": "PASS_FRESH_OWNER_SCOPED_REPAIR_DIAGNOSTIC", "scope": SCOPE, "model_constructors_at_admission": 0, "optimizer_updates_at_admission": 0, "real_world_outcomes": False, "independent_qualification": False, "promotion_authorized": False, "protected_TEST_authorized": False, "old_one_off_exception_transferred": False, "cryptographic_attestation_claimed": False}
    except (PermissionError, KeyError, FileNotFoundError) as error:
        return {"status": "BLOCKED_REPAIR_TRIAL", "reason": str(error), "model_constructors": 0, "optimizer_updates": 0, "old_one_off_exception_transferred": False}


def enforce(root, protocol, grant, output):
    receipt = inspect(root, protocol, grant)
    with (Path(output)/"ADMISSION_RESULT.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    if receipt["status"] != "PASS_FRESH_OWNER_SCOPED_REPAIR_DIAGNOSTIC":
        raise PermissionError(receipt["reason"])
    return receipt
