#!/usr/bin/env python3
"""Strict existing snapshot parity plus independently pinned new candidates."""
import json
import re
import sys
import urllib.request
from pathlib import Path

LOCAL = Path(__file__).resolve().parents[2] / "research/pnds-variant-math-links.json"
REMOTE = ("https://raw.githubusercontent.com/FPC-effortless/pnds-lean/"
          "main/"
          "foundation/v1/variant-math-links.json")

# A prospective paired draft must be compared with its reviewed immutable
# mathematics commit, not an unrelated old default-branch snapshot.
PIN = LOCAL.parents[1] / "research/langcap103/PNDS_PIN.json"
if PIN.exists():
    pin = json.loads(PIN.read_text())
    if pin.get("repository") != "FPC-effortless/pnds-lean" or not re.fullmatch("[0-9a-f]{40}", pin.get("commit", "")):
        raise SystemExit("Invalid immutable PNDS peer pin")
    REMOTE = ("https://raw.githubusercontent.com/FPC-effortless/pnds-lean/" +
              pin["commit"] + "/foundation/v1/variant-math-links.json")


def compare_existing_snapshot(local, remote):
    # LANGCAP103 and the original 17 rows keep their exact existing comparison.
    # Only the new candidate array uses its own independently pinned peer.
    left, right = dict(local), dict(remote)
    candidates = left.pop("prospective_candidates", [])
    right.pop("prospective_candidates", None)
    if left != right:
        raise ValueError("TAC-OSM and PNDS existing mathematical registries have drifted")
    if (not isinstance(candidates, list) or
            any(not isinstance(row, dict) for row in candidates)):
        raise ValueError("invalid prospective candidate registry")
    return candidates


def verify():
    local = json.loads(LOCAL.read_text(encoding="utf-8"))
    with urllib.request.urlopen(REMOTE, timeout=30) as response:
        remote = json.loads(response.read().decode("utf-8"))
    candidates = compare_existing_snapshot(local, remote)
    root = LOCAL.parents[1]
    candidate_pin = root / "research/langcache104/PNDS_PIN.json"
    if candidate_pin.exists() or candidates:
        if (len(candidates) != 1 or
                candidates[0].get("id") != "LANGCACHE104-DEEP-ASSOCIATIVE-CARRIED-CACHE"):
            raise ValueError("every new candidate needs its own reviewed pinned-peer verifier")
        sys.path.insert(0, str(root))
        from research.langcache104.verify_math_pins import verify as verify_candidate
        verify_candidate()
    print(json.dumps({"existing_snapshot_equal": True,
                      "separately_verified_prospective_candidates": len(candidates),
                      "cross_repository_linked_variants": len(local["variants"]),
                      "whole_model_source_equivalence_established": False}))


if __name__ == "__main__":
    verify()
