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


def fetch_bytes(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("immutable peer response exceeds bound")
    return raw


def verify_ocr004(base, read_remote):
    """Run the separate strict contract; a pass cannot waive legacy gates."""
    import importlib.util

    contract_root = Path(base) / "research/tac_ocr004"
    pin = json.loads((contract_root / "IMMUTABLE_PEER_PIN.json").read_text())
    spec = importlib.util.spec_from_file_location(
        "ocr004_native_peer_contract", contract_root / "verify_immutable_peer.py")
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    result = contract.verify_immutable_peer(base, "tac", pin, read_remote)
    # This is an intermediate scoped receipt. Overall success is emitted only
    # after every original legacy and non-OCR004 check below still succeeds.
    print(json.dumps({"ocr004_scoped_byte_correspondence": result,
                      "overall_canonical_gate_established": False}, sort_keys=True))
    return result

def verify():
    root = LOCAL.parents[1]
    verify_ocr004(root, fetch_bytes)
    local = json.loads(LOCAL.read_text(encoding="utf-8"))
    with urllib.request.urlopen(REMOTE, timeout=30) as response:
        remote = json.loads(response.read().decode("utf-8"))
    candidates = compare_existing_snapshot(local, remote)
    root = LOCAL.parents[1]
    candidate_pin = root / "research/langcache104/PNDS_PIN.json"
    ocr_pin = root / "research/tac_ocr001/PNDS_PIN.json"
    expected = set()
    if candidate_pin.exists():
        expected.add("LANGCACHE104-DEEP-ASSOCIATIVE-CARRIED-CACHE")
    if ocr_pin.exists():
        expected.update(("TAC-OCR-001-TAC", "TAC-OCR-001-TRANSFORMER", "TAC-OCR-001-GRU"))
    expected.update(("TAC-OCR-004-TAC", "TAC-OCR-004-TRANSFORMER", "TAC-OCR-004-GRU"))
    actual = [row.get("id") for row in candidates]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("every new candidate needs its own reviewed pinned-peer verifier")
    sys.path.insert(0, str(root))
    if candidate_pin.exists():
        from research.langcache104.verify_math_pins import verify as verify_candidate
        verify_candidate()
    if ocr_pin.exists():
        from research.tac_ocr001.verify_math_pins import verify as verify_ocr
        verify_ocr()
    print(json.dumps({"existing_snapshot_equal": True,
                      "separately_verified_prospective_candidates": len(candidates),
                      "cross_repository_linked_variants": len(local["variants"]),
                      "whole_model_source_equivalence_established": False}))


if __name__ == "__main__":
    verify()
