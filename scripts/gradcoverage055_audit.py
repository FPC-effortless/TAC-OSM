#!/usr/bin/env python3
"""Strict complete-gradient audit of unchanged original053/054 computation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tac_osm.research_gradient_coverage import GradientContract
from scripts.langrecomp052_replay import load_original
from scripts.langrecomp053_training_parity import GEOMETRY, RNG_SEED, compare_outputs, source_batch
from scripts.langrecomp054_warmed_scoped import POLICIES, execute
from scripts.langlayer035_sampling import fit_sources


def run(original_dir: Path) -> dict:
    if torch.cuda.is_initialized():
        raise ValueError("CPU-only audit; GPU not authorized")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    fit = fit_sources()
    model, receipt = load_original(original_dir, fit)
    model.train()
    contract = GradientContract.capture(model)
    if len(contract.trainable) != 55 or contract.elements != 499623:
        raise ValueError("historical original model declaration drift")
    rows = []
    for batch, length in GEOMETRY:
        x, y, owners, digest = source_batch(fit, batch, length)
        rng = torch.manual_seed(RNG_SEED).get_state().clone()
        anchor = None
        anchor_rng = None
        for policy in POLICIES:
            torch.set_rng_state(rng.clone())
            result = execute(model, x, y, policy)
            coverage = contract.validate(model, result, logits_shape=(batch, length, 256))
            if anchor is None:
                anchor = result
                anchor_rng = torch.get_rng_state().clone()
            comparison = compare_outputs(anchor, result)
            if not torch.equal(torch.get_rng_state(), anchor_rng):
                raise ValueError("original RNG differs between policies")
            # Deliberately corrupt only copied evidence, never weights/source/receipts.
            omitted = dict(result)
            omitted["grads"] = dict(result["grads"])
            first = next(iter(omitted["grads"]))
            del omitted["grads"][first]
            omitted["missing"] = [first]
            legacy_accepts = compare_outputs(omitted, omitted)
            try:
                contract.validate(model, omitted, logits_shape=(batch, length, 256))
            except ValueError:
                rejection = "PASS_REJECTED_COMMON_MISSING_GRADIENT"
            else:
                raise ValueError("strict coverage accepted omitted actual gradient")
            rows.append({"batch": batch, "window": length, "policy": policy,
                         "owners": owners, "source_windows_sha256": digest,
                         "coverage": coverage, "parity": comparison,
                         "common_omission_control": rejection,
                         "omitted_parameter": first,
                         "legacy_comparator_accepted_partial_elements": legacy_accepts["gradient_element_count"],
                         "rng_identical": True})
    contract.verify_model(model)
    source_paths = ["scripts/langrecomp053_training_parity.py", "scripts/langrecomp054_warmed_scoped.py",
                    "scripts/gradcoverage055_audit.py", "src/tac_osm/research_gradient_coverage.py"]
    return {"study": "GRADCOVERAGE055", "execution_level": "internal_research",
            "status": "PASS_COMPLETE_ORIGINAL_SOURCE_GRADIENT_COVERAGE",
            "original_checkpoint": receipt, "rows": rows,
            "live_declared_tensor_count": len(contract.trainable),
            "live_declared_element_count": contract.elements,
            "torch": torch.__version__, "cpu_threads": 2,
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths},
            "state_sha256": dict(contract.state_digests),
            "training_updates": 0, "optimizer_updates": 0, "gpu_minutes": 0,
            "protected_TEST": "NOT_ACCESSED", "independently_qualified": False,
            "promotion": False, "PNDS_full_source_AD_IEEE": "OPEN",
            "quality_memory_latency_evaluation": "NOT_RUN",
            "limits": ["Previously exposed original FIT and one frozen seed only",
                       "Gradient presence and equality do not prove correct mathematical derivatives or useful learning",
                       "No new model, fit, 100k-scale exception or prospective training authorization",
                       "Constructed common-omission controls are engineering evidence, not real predictive gains"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("do not overwrite evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args.original_dir)
    except Exception as exc:
        args.output.write_text(json.dumps({"study": "GRADCOVERAGE055", "status": "FAIL_OR_INCONCLUSIVE",
                                          "error_type": type(exc).__name__, "error": str(exc)}) + "\n")
        raise
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "actual_passes": len(result["rows"]),
                      "tensors_per_pass": result["live_declared_tensor_count"],
                      "elements_per_pass": result["live_declared_element_count"]}))


if __name__ == "__main__":
    main()
