#!/usr/bin/env python3
"""TAO002 source-authenticated optimizer preflight only: NO model fit.

This script *never* invokes backward(), optimizer.step(), accesses DEV, or
reads a protected cohort. It verifies the original LANGMEM086 100521p graph,
the licensed original FIT bytes, same-seed 160-step sample schedules, and
explicit FFN-only routing before writing a no-fit source receipt.

It does NOT grant authority to train, claim optimizer superiority, or
substitute for FAIR_COMPARISON preflight/final gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "src"),
    str(ROOT / "research/mathematical-engineering/langmem086"),
    str(ROOT / "research/mathematical-engineering/langbase024"),
]

import torch

from compact_tac import CompactDualClippedTAC086
from scripts.langmem086_run import fit_sources086, schedule, weight_hash
from tac_osm.optimizer_tao import TAO, build_tao

SEEDS = (1701, 1702, 1703, 1704, 1705)
ARMS = ("adamw", "muon", "adamuon", "normuon", "pre_ns")
ROUTE = ("ffn_up.weight",)
SOURCE_SHA = "911981cc63fd5436e0495cb067784e53475971131c811b7026e07d97a886d2d4"
MODEL_SHA = "7179908fa3672879f3c2046ec4521f0b9f783d5c696aaba82c4c2cd0b6ebc9b1"
SIZE = 100521
REG = ROOT / "research/mathematical-engineering/langmem086"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_original_graph() -> dict:
    model_path = REG / "compact_tac.py"
    if sha(model_path.read_bytes()) != MODEL_SHA:
        raise ValueError("original LANGMEM086 source graph hash changed")
    model = CompactDualClippedTAC086()
    if model.counts()["trainable"] != SIZE or model.counts()["deployed"] != SIZE:
        raise ValueError("actual 100521p active/deployed model size differs")
    named = dict(model.named_parameters())
    if tuple(n for n in named if n in ROUTE) != ROUTE:
        raise ValueError("unrecognized or duplicated FFN route")
    if named["ffn_up.weight"].shape != (94, 160):
        raise ValueError("FFN weight shape differs")
    if named["embedding.weight"].shape != (256, 32):
        raise ValueError("source vocab or encoder has changed")
    names = []
    for arm in ARMS:
        torch.manual_seed(1701)
        original = CompactDualClippedTAC086()
        before = weight_hash(original)
        if arm == "adamw":
            opt = torch.optim.AdamW(original.parameters(), lr=0.001, weight_decay=0.01)
            orthogonal = []
        else:
            opt = build_tao(original, orthogonal_names=ROUTE, mode=arm, lr=0.001,
                            weight_decay=0.01)
            group = next(g for g in opt.param_groups if g["kind"] == "orthogonal")
            orthogonal = [n for n, p in original.named_parameters()
                          if any(p is q for q in group["params"])]
            if tuple(orthogonal) != ROUTE:
                raise ValueError("accidental matrix routing")
            if not isinstance(opt, TAO):
                raise ValueError("unexpected optimizer implementation")
        if weight_hash(original) != before:
            raise ValueError("optimizer construction modified initial weights")
        names.append({"arm": arm, "initial_hash_seed1701": before,
                      "orthogonal_parameter_names": orthogonal,
                      "orthogonal_parameter_count": sum(named[n].numel() for n in orthogonal)})
    if len({x["initial_hash_seed1701"] for x in names}) != 1:
        raise ValueError("same-seed initialization differs by optimizer")
    return {"model_source_sha256": MODEL_SHA, "parameter_count": SIZE,
            "eligible_ffn_shape": [94, 160], "routing": names}


def audit_fit_samples() -> dict:
    docs, source = fit_sources086()
    if source != SOURCE_SHA or len(docs) != 14:
        raise ValueError("unapproved original publisher source or FIT owner count")
    original = json.loads((REG / "FIT_PLAN.json").read_text())
    if original["fit_source_sha256"] != source:
        raise ValueError("frozen source checksum drift")
    results = []
    for seed in SEEDS:
        s = schedule(docs, seed)
        if s["steps"] != 160 or s["batch"] != 4 or s["context"] != 256:
            raise ValueError("source sample schedule changed")
        if seed in (1701, 1702, 1703):
            old = next(x for x in original["by_seed"] if x["seed"] == seed)
            if s != old:
                raise ValueError("historical schedule prefix changed")
        results.append(s)
    return {"original_FIT_sha256": source,
            "fit_owner_count": len(docs),
            "frozen_sample_schedules": results,
            "source_license": "PSF License v2; documentation code examples 0BSD",
            "source_release_commit": "60403a5409ff2c3f3b07dd2ca91a7a3e096839c7"}


def prefit() -> dict:
    torch.set_num_threads(2)
    graph = verify_original_graph()
    samples = audit_fit_samples()
    return {"study": "TAO002", "phase": "SOURCE_AND_GRAPH_NO_FIT_PREFLIGHT",
            "scope": "internal_research", "fit_performed": False,
            "dev_bytes_accessed": False, "protected_TEST_accessed": False,
            "optimizer_steps": 0, "requested_fit_authorized_by_this_receipt": False,
            "full_matched_comparison_gate": "NOT_YET_REGISTERED_OR_EXECUTED",
            "original_checkpoint_replay_gate": "NOT_YET_RUN",
            "graph": graph, "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Optional JSON receipt; does not fit any model")
    args = parser.parse_args()
    result = prefit()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        dest = Path(args.output)
        if dest.exists():
            raise FileExistsError("refusing to overwrite prior prefit evidence")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
