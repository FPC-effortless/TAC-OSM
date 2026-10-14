"""DEV-015: fail-closed checkpoint, source and full episode evaluation replay.

Evaluation replay reconstructs saved predictions from checkpoints; it does
NOT reproduce optimizer training, constitute an independent external holdout,
or certify a production TAC-OSM architecture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from tac_osm.development.noisy_memory import parameter_hash
from tac_osm.development.plm_casm_bridge import ARMS,PLMCASMBridge
from run_plm_casm_bridge_dev import (
    ROOT,TRAIN_SEEDS,EVAL_SEEDS,PARENT_SHA256,
    read_parent_report,open_parent,evaluate,summarize,
)


def validate(parent_archive:Path,folder:Path):
    parent_report=read_parent_report(parent_archive)
    report=json.loads((folder/"report.json").read_text())
    if report.get("study")!="DEV-015" or (
            report.get("status")!="DEVELOPMENT_ONLY_NOT_CONFIRMATORY"
            or report.get("parent_artifact_sha256")!=PARENT_SHA256):
        raise AssertionError("wrong study, promotion or parent")
    seeds=TRAIN_SEEDS[:1] if report.get("smoke") else TRAIN_SEEDS
    if report.get("seeds")!=list(seeds) or report.get("arms")!=list(ARMS):
        raise AssertionError("incomplete evaluation arms")
    if len(report.get("records",[]))!=len(seeds):
        raise AssertionError("missing seed")
    for f,digest in report["source_sha256"].items():
        if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=digest:
            raise AssertionError("source changed since experiment: "+f)
    episodes=0
    for seed,entry in zip(seeds,report["records"]):
        evseed=EVAL_SEEDS[TRAIN_SEEDS.index(seed)]
        if entry["seed"]!=seed or entry["eval_seed"]!=evseed:
            raise AssertionError("wrong eval seed mapping")
        if set(entry["models"])!=set(ARMS) or set(entry["rows"])!=set(ARMS):
            raise AssertionError("incomplete model grid")
        meta=entry["training"]
        if set(meta["initial_hashes"])!=set(ARMS) or len(
                set(meta["initial_hashes"].values()))!=1:
            raise AssertionError("initialization mismatch")
        if len(set(meta["train_fingerprints"]))!=4*meta["steps"]:
            raise AssertionError("missing or duplicate training events")
        train=set(meta["train_fingerprints"])
        parent=open_parent(parent_archive,parent_report,seed)
        parenthash=parameter_hash(parent)
        if meta["frozen_parent_hash"]!=parenthash:
            raise AssertionError("parent tensor mismatch")
        for arm in ARMS:
            info=entry["models"][arm]
            name=f"{arm}-seed{seed}.pt"
            if info["file"]!=name or Path(name).name!=name:
                raise AssertionError("unexpected checkpoint filename")
            p=folder/name
            if hashlib.sha256(p.read_bytes()).hexdigest()!=info["sha256"]:
                raise AssertionError("modified checkpoint")
            m=PLMCASMBridge(parent,mode=arm,top_k=1)
            m.load_state_dict(torch.load(p,map_location="cpu",weights_only=True),
                              strict=True)
            m.eval()
            if parameter_hash(m)!=meta["final_hashes"][arm] or (
                    parameter_hash(m)==meta["initial_hashes"][arm]):
                raise AssertionError("model weights invalid")
            if sum(x.numel() for x in m.parameters())!=info["total_parameters"]:
                raise AssertionError("capacity mismatched")
            if sum(x.numel() for x in m.parameters() if x.requires_grad)!=info["trainable_parameters"]:
                raise AssertionError("trainable parameter count changed")
            saved=entry["rows"][arm]
            registered=evaluate(m,evseed,report["smoke"])
            if registered!=saved:
                raise AssertionError(f"evaluation changed seed={seed} arm={arm}")
            if summarize(saved)!=entry["summaries"][arm]:
                raise AssertionError("metrics not derivable from raw rows")
            if len(saved)!=(2 if report["smoke"] else 160):
                raise AssertionError("missing OOD or in-family evaluation cell")
            if train & {f for row in saved for f in row["fingerprints"]}:
                raise AssertionError("training/eval fingerprint overlap")
            if any(row["plm_scalars_per_episode"]!=768 for row in saved):
                raise AssertionError("unreported memory capacity")
            if any(row["coordinate_family"] not in ("original","signed_permuted")
                   for row in saved):
                raise AssertionError("wrong shift control")
            if any(entry["summaries"][arm][family]["no_bit"]!=.5
                   for family in ("original","signed_permuted")):
                raise AssertionError("no-bit test leakage")
            episodes+=2*len(saved)
        if parameter_hash(parent)!=parenthash:
            raise AssertionError("parent mutated")
    return {
        "status":"REPLAY_VALIDATED_DEVELOPMENT_ONLY",
        "checkpoints":len(seeds)*len(ARMS),
        "replayed_action_episodes":episodes,
        "warning":"Evaluation checkpoint replay ONLY; no SGD or independent replication.",
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--parent",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(validate(a.parent,a.output),indent=2))
