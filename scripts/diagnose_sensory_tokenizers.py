#!/usr/bin/env python3
"""No-fit source-only tokenizer diagnosis; deliberately no model accuracy score."""
from __future__ import annotations
import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path
from time import perf_counter_ns

# Allow running from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tac_osm.sensory_tokenizers import VARIANTS, tokenize


FIXTURE = b"The motor fault appeared at 12:30. Motor restarted.\n" * 20


def diagnose(data: bytes, query: bytes, *, provenance: str, split: str) -> dict:
    rows = []
    for variant in VARIANTS:
        args = {"variant": variant}
        if variant in ("relevance", "hybrid"):
            args["query"] = query
        start = perf_counter_ns()
        result = tokenize(data, **args)
        elapsed = perf_counter_ns() - start
        rows.append({
            "variant": variant,
            "lossless_reconstruction": result.reconstruct() == data,
            "patches": len(result.patches),
            "leaf_nodes": len(result.leaves),
            "graph_edges": len(result.edges),
            "total_representation_nodes": result.total_representation_nodes,
            "bytes_per_patch": round(len(data) / len(result.patches), 5) if result.patches else None,
            "scan_work_units_lower_bound": result.scan_work,
            "wall_time_ns_uncontrolled": elapsed,
            "approx_payload_bytes_excluding_metadata": sum(len(p.payload) for p in result.patches),
            "metadata_and_raw_byte_storage_excluded_from_token_count": True,
            "streaming_prefix_stability": "tested for committed patches" if variant in ("fixed", "entropy", "sensory") else "UNKNOWN_offline_only",
            "casm_execution": False,
            "cdl_teacher_scoring": False,
        })
    return {
        "study_id": "SENSE-TOK-001",
        "evidence_class": "NO_FIT_ENGINEERING_DIAGNOSTIC_ONLY",
        "source_provenance": provenance,
        "split": split,
        "data_sha256": sha256(data).hexdigest(),
        "query_sha256": sha256(query).hexdigest(),
        "input_bytes": len(data),
        "trained_models": 0,
        "capability_comparison_qualified": False,
        "primary_real_world_evaluation": False,
        "source_authentication": "NOT_VERIFIED_BY_THIS_SCRIPT",
        "scientific_parity": "INCONCLUSIVE_NOT_APPLICABLE_TO_NO_FIT",
        "pnds_formalization": "OPEN",
        "notes": "Fewer patches do not establish useful compression or model efficiency; raw bytes and sidecar structures remain. Synthetic fixture has zero capability evidentiary weight.",
        "variants": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixture", action="store_true", help="explicitly constructed fixture, no capability evidence")
    g.add_argument("--input", type=Path, help="operator-supplied locally authorized FIT/DEV data file")
    ap.add_argument("--split", choices=("FIT", "DEV"), default="FIT")
    ap.add_argument("--source", help="publisher/version/licence provenance for a real source")
    ap.add_argument("--query", default="motor fault", help="predecision query only; never a label/answer")
    ap.add_argument("--max-bytes", type=int, default=1_000_000)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if not (0 < args.max_bytes <= 10_000_000):
        ap.error("max-bytes must be in 1..10,000,000")
    if args.input is not None:
        if not args.source:
            ap.error("--source publisher/version/licence required for real input")
        # No symlink / special-file / huge upload or unbounded allocations.
        if not args.input.is_file() or args.input.is_symlink():
            ap.error("input must be a regular, non-symlink file")
        with args.input.open("rb") as fh:
            data = fh.read(args.max_bytes + 1)
        if len(data) > args.max_bytes:
            ap.error("input exceeds max-bytes; no silent truncation")
        provenance = args.source
    else:
        data, provenance = FIXTURE, "GENERATED_FIXTURE_NON_EMPIRICAL"
    report = diagnose(data, args.query.encode("utf8"), provenance=provenance, split=args.split)
    payload = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
