#!/usr/bin/env python3
"""SENSE-TOK-006 no-fit readiness: --check-training exits 2 until qualified.

Instantiates candidates and reports honest blockers; never constructs an
optimizer, reads any corpus/TEST, signs evidence or submits training jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tac_osm.adaptive_token_controls import ARM_NAMES, TokenComparisonArm
from tac_osm.phone_research_ops import decide, load_proposal
from scripts.validate_research_parity import check_manifest


def readiness(root: Path = ROOT) -> dict:
    protocol_path = root / "research/sensory_tokenizers/SENSE-TOK-006-TRAINING-PROTOCOL.json"
    raw = protocol_path.read_bytes()
    protocol = json.loads(raw)
    if (protocol.get("arms") != list(ARM_NAMES)
            or protocol.get("seeds") != [1, 2, 3, 4, 5]
            or protocol.get("training_authorized") is not False
            or protocol.get("no_new_model_fits") is not True):
        raise ValueError("draft contract changed; full reviewed training registration required")
    # .5 is an instantiated PARAMETER-COUNT fixture, not real FIT calibration.
    # It is never approved for an experiment or used to produce prediction scores.
    candidates = [TokenComparisonArm(name, seed=1, random_byte_probability=.5
                  if name == "random_gate" else None).parameter_receipt()
                  for name in ARM_NAMES]
    gaps = {key: (max(r[key] for r in candidates) - min(r[key] for r in candidates))
                 / max(r[key] for r in candidates)
            for key in ("trainable_parameters", "deployed_parameters")}
    if any(not r["within_100k_band"] for r in candidates) or max(gaps.values()) > .01:
        raise ValueError("candidate parameter contract failed")
    manifest = json.loads((root / "research/mathematical-engineering/sense_tok_006/FAIR_COMPARISON.json").read_bytes())
    diagnostic = check_manifest(manifest, "preflight")
    if diagnostic["status"] != "PASS_DIAGNOSTIC_ONLY":
        raise ValueError("this command may only run the no-fit preparation registration")
    plan = decide(load_proposal(Path("research/phone_queue/sense-tok-006-readiness.json"), root))
    gates = protocol.get("open_gates")
    if not isinstance(gates, dict) or not gates or any(value != "OPEN" for value in gates.values()):
        raise ValueError("source-controlled status labels cannot certify external gates")
    return {"study_id": protocol["study_id"], "status": "BLOCKED_BEFORE_FIRST_FIT",
            "evidence_class": "NO_FIT_COMPARISON_PREPARATION",
            "protocol_sha256": hashlib.sha256(raw).hexdigest(),
            "source_parent_commit": protocol["model_ancestor_commit"],
            "source_file_sha256": {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                for path in ("src/tac_osm/adaptive_byte_token.py",
                             "src/tac_osm/adaptive_codebook_token.py",
                             "src/tac_osm/adaptive_token_controls.py",
                             "src/tac_osm/adaptive_token_evaluation.py",
                             "scripts/prepare_sense_tok_learning.py")},
            "candidate_parameters": candidates, "max_relative_parameter_gaps": gaps,
            "random_probability_status": "UNQUALIFIED_COUNT_ONLY_FIXTURE",
            "diagnostic_registration": diagnostic, "compute_plan": plan,
            "open_gates": gates, "model_fits": 0, "optimizer_steps": 0,
            "test_accessed": False, "training_authorized": False,
            "note": "No optimizer exists in this command; a diagnostic PASS cannot authorize a fit."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-training", action="store_true")
    args = parser.parse_args(argv)
    report = readiness()
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 2 if args.check_training else 0


if __name__ == "__main__":
    raise SystemExit(main())
