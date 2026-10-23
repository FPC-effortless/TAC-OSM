"""Verify immutable registrations/checkpoints; never authorize optimizer updates."""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tac_osm.parameter_budget import ParameterBudgetError, canonical_sha

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research/governance/FROZEN_REPLAY_REGISTRY.json"


def git_bytes(commit, path):
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ParameterBudgetError("frozen replay requires a full pinned commit")
    if Path(path).is_absolute() or ".." in Path(path).parts:
        raise ParameterBudgetError("invalid registry path")
    result = subprocess.run(
        ["git", "cat-file", "blob", commit + ":" + path], cwd=ROOT, capture_output=True
    )
    if result.returncode:
        raise ParameterBudgetError(
            "frozen Git object unavailable; fetch the exact pinned commit first"
        )
    return result.stdout


def verify_frozen_replay(manifest, key):
    if not isinstance(key, str):
        raise ParameterBudgetError(
            "explicit immutable replay provenance required; old study names are insufficient"
        )
    registry = json.loads(REGISTRY.read_text())
    entry = registry["entries"].get(key)
    if entry is None:
        raise ParameterBudgetError("unrecognized frozen replay provenance")
    raw = git_bytes(entry["commit"], entry["registration_path"])
    if hashlib.sha256(raw).hexdigest() != entry["registration_sha256"] or canonical_sha(
        json.loads(raw)
    ) != canonical_sha(manifest):
        raise ParameterBudgetError(
            "registration differs from frozen historical provenance"
        )
    if not entry.get("checkpoints"):
        raise ParameterBudgetError(
            "frozen replay requires original repository checkpoint provenance"
        )
    for row in entry["checkpoints"]:
        b = git_bytes(entry["commit"], row["path"])
        if len(b) != row["bytes"] or hashlib.sha256(b).hexdigest() != row["sha256"]:
            raise ParameterBudgetError("historical checkpoint identity mismatch")
    return {
        "status": "PASS_FROZEN_REPLAY_ONLY",
        "original_commit": entry["commit"],
        "verified_original_checkpoints": len(entry["checkpoints"]),
        "training_authorized": False,
        "eligible_for_architecture_superiority": False,
        "scope": "Integrity of original Git registration/checkpoint bytes; prediction quality and score replay are separate.",
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        required=True,
        help="audit every registered immutable snapshot; never fit models",
    )
    parser.parse_args()
    registry = json.loads(REGISTRY.read_text())
    results = {}
    for key, entry in registry["entries"].items():
        manifest = json.loads(git_bytes(entry["commit"], entry["registration_path"]))
        results[key] = verify_frozen_replay(manifest, key)
    print(
        json.dumps(
            {"status": "PASS", "training_authorized": False, "entries": results},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
