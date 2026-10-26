#!/usr/bin/env python3
"""Guarded TAC-OSM prospective training launcher: common to ALL architectures.

  python scripts/run_qualified_research.py \
    --prefit-manifest frozen.json --final-manifest measured.json \
    --prefit-evidence signed-prefit.json --final-evidence signed-postfit.json \
    -- python scripts/MY_ACTUAL_AUTHORIZED_TRAINING_RUNNER.py ...

No model fitting is performed by this wrapper itself. No shell expansion.
The operator supplies the independently pinned public key in
TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX (never supplied via the actor manifest).
Postfit evidence must be generated independently AFTER genuine outcomes;
this wrapper cannot manufacture or sign it. No key = no training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tac_osm.research_reward_runner import binding, verify_stage_continuity  # noqa: E402
from scripts.validate_research_parity import check_manifest  # noqa: E402


def _read(path: Path):
    return json.loads(path.read_bytes())


def _registration_without_measured(manifest: dict) -> dict:
    """Only actual resource/outcome measurement receipts may be added postfit.

    Budget, source, splits, model names, seeds, policies and optimizer design
    cannot be changed after the first training decision.
    """
    frozen = json.loads(json.dumps(manifest, sort_keys=True, allow_nan=False))
    rows = frozen.get("models") if frozen.get(
        "study_type") == "single_model_training" else frozen.get("arms")
    if not isinstance(rows, list):
        raise ValueError("invalid registered model arm list")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid model arm")
        row.pop("measured", None)
    return frozen


# Actor subprocess receives no GitHub, cloud, evaluator, training-coordinator,
# source-secret or signing credentials from the invoking audit runner. Genuine
# source access must be separately granted with least privilege by an isolated
# actor environment, never inherited from evaluator.
_ACTOR_ENV_ALLOWLIST = frozenset((
    "PATH", "PYTHONPATH", "VIRTUAL_ENV", "HOME", "LANG", "LC_ALL",
    "TMPDIR", "TEMP", "TMP", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "ATEN_CPU_CAPABILITY", "CUDA_VISIBLE_DEVICES",
))


def guarded_execute(args) -> dict:
    public = os.environ.get("TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX")
    if not public:
        raise ValueError("trusted operator public key missing; optimizer forbidden")
    raw = args.prefit_manifest.read_bytes()
    frozen = json.loads(raw)
    prefit = check_manifest(
        frozen, "preflight",
        reward_attestation=_read(args.prefit_evidence),
        trusted_reward_public_key_hex=public)
    if prefit["reward_integrity"]["status"] != "PASS_SIGNED_EXTERNAL_AUDIT":
        raise ValueError("prefit reward gate not qualified")
    command = args.command
    if not command or command[0] != "--" or len(command) < 2:
        raise ValueError("explicit -- command required; shell invocation forbidden")
    executable = command[1:]
    # Default-deny: inherited CI/host/cloud variables may contain signing,
    # evaluator, source, repository or scoring credentials. No model receives
    # any of them through this launcher.
    clean_env = {key: value for key, value in os.environ.items()
                 if key in _ACTOR_ENV_ALLOWLIST}
    finished = subprocess.run(executable, check=False, shell=False, env=clean_env,
                              cwd=ROOT)
    if finished.returncode:
        raise ValueError("trained runner failed; no model promotion")
    if hashlib.sha256(args.prefit_manifest.read_bytes()).digest() != hashlib.sha256(raw).digest():
        raise ValueError("frozen prefit registration changed during training")
    final = _read(args.final_manifest)
    if binding(final) != binding(frozen):
        raise ValueError("postfit model/source/seed set differs from frozen preflight")
    if _registration_without_measured(final) != _registration_without_measured(frozen):
        raise ValueError("postfit preregistration changed outside actual measured receipts")
    post = check_manifest(
        final, "final",
        reward_attestation=_read(args.final_evidence),
        trusted_reward_public_key_hex=public)
    if post["reward_integrity"]["status"] != "PASS_SIGNED_EXTERNAL_AUDIT":
        raise ValueError("postfit reward gate not qualified")
    verify_stage_continuity(_read(args.prefit_evidence), _read(args.final_evidence))
    return {"status": "PASS_SIGNED_PREFIT_AND_POSTFIT",
            "study_id": frozen["study_id"],
            "registered_variant_seeds": len(binding(frozen)["variant_seeds"]),
            "attestation_provenance": "externally_signed_evaluator_receipts",
            "qualification": "signed_evidence_gate_not_proof_of_causal_world_model_success"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefit-manifest", required=True, type=Path)
    parser.add_argument("--final-manifest", required=True, type=Path)
    parser.add_argument("--prefit-evidence", required=True, type=Path)
    parser.add_argument("--final-evidence", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        result = guarded_execute(args)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, KeyError, TypeError, OSError,
            json.JSONDecodeError, RuntimeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": type(exc).__name__,
                          "qualification": "no_training_or_promotion_claim"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
