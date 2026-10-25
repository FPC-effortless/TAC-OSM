#!/usr/bin/env python3
"""CI peer-snapshot parity: TAC-OSM reciprocal links must equal PNDS links."""
import json
import urllib.request
from pathlib import Path

LOCAL = Path(__file__).resolve().parents[2] / "research/pnds-variant-math-links.json"
REMOTE = ("https://raw.githubusercontent.com/FPC-effortless/pnds-lean/"
          "f05d88d6e581e56255d25a3d45fa68d1d202995d/"
          "foundation/v1/variant-math-links.json")


def verify():
    local = json.loads(LOCAL.read_text(encoding="utf-8"))
    with urllib.request.urlopen(REMOTE, timeout=30) as response:
        remote = json.loads(response.read().decode("utf-8"))
    if local != remote:
        raise SystemExit("TAC-OSM and PNDS mathematical version registries have drifted")
    print(json.dumps({"reciprocal_manifest_equal": True,
                      "cross_repository_linked_variants": len(local["variants"]),
                      "whole_model_source_equivalence_established": False}))


if __name__ == "__main__":
    verify()
