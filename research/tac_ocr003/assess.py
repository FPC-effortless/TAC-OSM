"""Verify actual checkpoint counts and the independent final attestation."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import torch

from research.tac_ocr001.model import GroundedOCRDecoder
from research.tac_ocr001.presets import matched_presets
from .core import MANIFEST, PROTOCOL, ROOT, sha256, write_json


def final_gate(output):
    sys.path.insert(0, str(ROOT / "src"))
    from scripts.validate_research_parity import check_manifest
    manifest = copy.deepcopy(json.loads(MANIFEST.read_text()))
    protocol = json.loads(PROTOCOL.read_text())
    for row in manifest["models"]:
        root = output / f"seed-{row['seed']}"
        completion = json.loads((root / "completion.json").read_text())
        value = torch.load(root / "last.pt", map_location="cpu", weights_only=True)
        if (completion["status"] != "FIT_COMPLETED_UNQUALIFIED" or
                value["cursor"] != {"epoch": protocol["epochs"], "batch": 0} or
                value["identity"]["protocol_sha256"] != sha256(PROTOCOL) or
                value["identity"]["manifest_sha256"] != sha256(MANIFEST) or
                value["identity"]["seed"] != row["seed"]):
            raise ValueError("actual checkpoint is incomplete or from another source/protocol/seed")
        model = GroundedOCRDecoder(matched_presets()["tac"])
        model.load_state_dict(value["model"], strict=True)
        encoder_counts = [json.loads((output / "features" / f"shard-{slot}.json").read_text())["encoder_parameters"]
                          for slot in range(2)]
        if encoder_counts != [protocol["frozen_encoder_parameters"]] * 2:
            raise ValueError("frozen encoder actual count differs")
        counts = model.counts(frozen_encoder_parameters=encoder_counts[0])
        digest = sha256(root / "last.pt")
        if digest != completion["last_checkpoint_sha256"]:
            raise ValueError("checkpoint changed after fit")
        row["measured"] = {"trainable_parameters": counts["trainable"],
                           "deployed_parameters": counts["total_deployed"],
                           "checkpoint_sha256": digest}
    write_json(output / "FAIR_COMPARISON_MEASURED.json", manifest)
    raw = os.environ.get("TAC_OSM_POSTFIT_ATTESTATION_JSON")
    report = check_manifest(manifest, "final", reward_attestation=json.loads(raw) if raw else None,
                           trusted_reward_public_key_hex=os.environ.get("TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX"))
    write_json(output / "independent-final-gate.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        final_gate(args.output)
    except Exception as exc:
        write_json(args.output / "independent-final-blocked.json", {
            "status": "POSTFIT_QUALIFICATION_BLOCKED", "exception_type": type(exc).__name__,
            "reason": str(exc)[:500], "qualified": False})
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
