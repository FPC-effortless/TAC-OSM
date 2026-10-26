#!/usr/bin/env python3
"""No-fit TAC-OSM research observatory (native-model CLI adapter).

Fixture (unit/engineering only):
  python scripts/run_research_observatory.py --fixture
Checkpoint on an independently verified FIT/DEV input export:
  python scripts/run_research_observatory.py --input-json source.json \
      --expected-input-sha256 HASH --checkpoint model.pt \
      --expected-checkpoint-sha256 HASH --run-id me-example
Compare paired original-run receipts (never opens sealed TEST itself):
  python scripts/run_research_observatory.py --compare left.json right.json

Other model types use research_probe.inspect_variant with an explicit, trusted\nadapter. No model fitting. No automatic heldout data access. No accuracy claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from tac_osm.predictive_state_agent import PredictiveStateAgent  # noqa: E402
from tac_osm.research_observatory import interpret_snapshot  # noqa: E402
from tac_osm.research_probe import inspect_variant  # noqa: E402
from tac_osm.research_probe_adapters import native_predictive_adapter  # noqa: E402
from tac_osm.research_evidence import compare_evidence  # noqa: E402


def pinned_bytes(path: Path, expected: str) -> bytes:
    if len(expected) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected):
        raise ValueError("a preregistered 64-digit SHA256 is required")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest().lower() != expected.lower():
        raise ValueError("input/checkpoint bytes do not match pinned SHA256")
    return payload


def snapshot(model, history, candidates, run_id, *, profile=False, isolation=False):
    """Native CLI compatibility wrapper; generic core executes every probe.

    Other architectures call inspect_variant() with their own trusted adapter.
    The CLI intentionally never guesses model types from an untrusted state_dict.
    """
    result = inspect_variant(
        model, {"history": history, "candidates": candidates},
        native_predictive_adapter, isolation=isolation, profile=profile)
    result["run_id"] = run_id
    result["gradient_conflicts"] = {
        "status": "NOT_RUN_NO_AUTHENTIC_SUPERVISED_LOSSES",
        "requirement": "pass genuine FIT/DEV loss tensors to gradient_conflict_atlas",
    }
    result["interpretation"] = interpret_snapshot(result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--fixture", action="store_true")
    group.add_argument("--input-json", type=Path)
    group.add_argument("--compare", nargs=2, type=Path)
    ap.add_argument("--expected-input-sha256")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--expected-checkpoint-sha256")
    ap.add_argument("--run-id", default="no-fit-diagnostic")
    ap.add_argument("--profile", action="store_true",
                    help="measure local replay-and-query timing on this host")
    ap.add_argument("--entity-isolation", action="store_true",
                    help="compare batched and per-entity forecasts (requires B>=2)")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    try:
        if args.compare:
            if args.checkpoint or args.input_json or args.fixture:
                raise ValueError("comparison mode reads only two receipts")
            left, right = (json.loads(path.read_text(encoding="utf-8"))
                           for path in args.compare)
            result = compare_evidence(left, right)
            exitcode = 0 if result["status"] == "RECEIPT_COMPATIBLE" else 1
        else:
            if args.fixture:
                if args.checkpoint:
                    raise ValueError("fixture cannot consume a checkpoint")
                torch.manual_seed(37)
                model = PredictiveStateAgent(5, 3, 1).eval()
                history = torch.randn(2, 4, 5)
                candidates = torch.randn(2, 3, 3)
                provenance = "constructed_untrained_fixture_no_capability"
            else:
                if not (args.checkpoint and args.expected_input_sha256
                        and args.expected_checkpoint_sha256):
                    raise ValueError("pinned input and checkpoint hashes required")
                raw = pinned_bytes(args.input_json, args.expected_input_sha256)
                doc = json.loads(raw)
                if not isinstance(doc, dict) or set(doc) != {"history", "candidates"}:
                    raise ValueError("input JSON must contain ONLY history and candidates")
                history = torch.tensor(doc["history"], dtype=torch.float32)
                candidates = torch.tensor(doc["candidates"], dtype=torch.float32)
                if history.ndim != 3 or candidates.ndim != 3:
                    raise ValueError("input tensor dimensionality invalid")
                pinned_bytes(args.checkpoint, args.expected_checkpoint_sha256)
                # weights_only rejects arbitrary Python object deserialization.
                state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
                if not isinstance(state, dict) or "readout.2.weight" not in state:
                    raise ValueError("checkpoint must be native PredictiveStateAgent state_dict")
                model = PredictiveStateAgent(
                    history.shape[-1], candidates.shape[-1],
                    state["readout.2.weight"].shape[0])
                model.load_state_dict(state, strict=True)
                model.eval()
                provenance = "caller_pinned_input_and_checkpoint_only_not_authenticated_publisher"
            result = snapshot(model, history, candidates, args.run_id,
                              profile=args.profile, isolation=args.entity_isolation)
            result["source_provenance"] = provenance
            exitcode = 0
        output = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        print(output, end="")
        return exitcode
    except (ValueError, TypeError, KeyError, RuntimeError, OSError,
            json.JSONDecodeError) as exc:
        print(json.dumps({"schema": "tac-osm-observatory/error-v1",
                          "status": "BLOCKED", "error": type(exc).__name__,
                          "detail": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
