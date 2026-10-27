#!/usr/bin/env python3
"""CI peer-snapshot parity: TAC-OSM reciprocal links must equal PNDS links."""
import argparse
import json
import re
import urllib.request
from pathlib import Path

LOCAL = Path(__file__).resolve().parents[2] / "research/pnds-variant-math-links.json"
REMOTE = ("https://raw.githubusercontent.com/FPC-effortless/pnds-lean/"
          "main/"
          "foundation/v1/variant-math-links.json")


def verify(peer_ref="main"):
    if peer_ref != "main" and not re.fullmatch(r"[0-9a-f]{40}",peer_ref):
        raise SystemExit("draft peer-ref must be an immutable commit")
    local = json.loads(LOCAL.read_text(encoding="utf-8"))
    with urllib.request.urlopen(REMOTE.replace("/main/", "/"+peer_ref+"/"), timeout=30) as response:
        remote = json.loads(response.read().decode("utf-8"))
    if local != remote:
        raise SystemExit("TAC-OSM and PNDS mathematical version registries have drifted")
    print(json.dumps({"reciprocal_manifest_equal": True,
                      "cross_repository_linked_variants": len(local["variants"]),
                      "whole_model_source_equivalence_established": False}))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--peer-ref",default="main")
    verify(p.parse_args().peer_ref)
