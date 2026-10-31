#!/usr/bin/env python3
"""CI peer-snapshot parity: TAC-OSM reciprocal links must equal PNDS links."""
import json
import re
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
