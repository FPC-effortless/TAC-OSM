"""DEV-014: replay all action decisions from pinned DEV-013 & CASM checkpoints.

This validates evaluation, not optimization or independent model replication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from tac_osm.development.noisy_memory import parameter_hash
from tac_osm.development.casm_action_transfer import CASMActionAdapter
from run_casm_action_transfer_dev import (
    ROOT,ARMS,TOPK,TRAIN_SEEDS,EVAL_SEEDS,PARENT_SHA256,
    read_parent_report,open_parent,generated_eval_cases,evaluate,summarize,
)


def validate(parent_archive:Path,out:Path):
    parent_report=read_parent_report(parent_archive)
    r=json.loads((out/"report.json").read_text())
    if (r.get("study")!="DEV-014" or
        r.get("status")!="DEVELOPMENT_ONLY_NOT_CONFIRMATORY" or
        r.get("parent_artifact_sha256")!=PARENT_SHA256):
        raise AssertionError("experiment/parent identity altered")
    seeds=TRAIN_SEEDS[:1] if r.get("smoke") else TRAIN_SEEDS
    if r.get("seeds")!=list(seeds) or r.get("arms")!=list(ARMS):
        raise AssertionError("missing seeds/arms")
    if len(r.get("records",[]))!=len(seeds):
        raise AssertionError("wrong number of records")
    for name,digest in r["source_sha256"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise AssertionError("source changed "+name)
    evaluated=0
    for record,seed in zip(r["records"],seeds):
        if record["seed"]!=seed or record["eval_seed"]!=EVAL_SEEDS[TRAIN_SEEDS.index(seed)]:
            raise AssertionError("wrong evaluation seed")
        if set(record["models"])!=set(ARMS) or set(record["rows"])!=set(ARMS):
            raise AssertionError("incomplete arms")
        parent=open_parent(parent_archive,parent_report,seed)
        parent_sha=parameter_hash(parent)
        if record["training"]["parent_tensor_hash"]!=parent_sha:
            raise AssertionError("frozen parent mismatched")
        initial=record["training"]["initial_hashes"]
        if len(set(initial.values()))!=1 or set(initial)!=set(ARMS):
            raise AssertionError("unmatched initial models")
        meta=record["training"]
        expected_train=12 if r["smoke"] else 480
        if meta["train_episodes"]!=expected_train or len(
                set(meta["train_fingerprints"]))!=expected_train:
            raise AssertionError("bad training stream")
        baseline_fingerprints=None
        for arm in ARMS:
            info=record["models"][arm]
            name=info["file"]
            if (name!=f"{arm}-seed{seed}.pt" or Path(name).name!=name or
                info["top_k"]!=TOPK[arm]):
                raise AssertionError("bad model checkpoint")
            path=out/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=info["sha256"]:
                raise AssertionError("checkpoint byte mismatch")
            model=CASMActionAdapter(parent,top_k=TOPK[arm])
            model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),
                                  strict=True)
            model.eval()
            if parameter_hash(model)!=meta["final_hashes"][arm] or (
                    parameter_hash(model)==initial[arm]):
                raise AssertionError("trained tensor mismatch")
            if sum(p.numel() for p in model.parameters())!=info["parameter_count"]:
                raise AssertionError("parameter mismatch")
            if not meta["real_casm_selector_gradient"][arm]:
                raise AssertionError("missing selector training gradient")
            saved=record["rows"][arm]
            registered=list(generated_eval_cases(record["eval_seed"],r["smoke"]))
            if len(saved)!=len(registered):
                raise AssertionError("missing evaluation cell")
            expected_conditions=[dict(zip(
                ("h","d","near","noise","role","rep"),c)) for c,_ in registered]
            if [row["condition"] for row in saved]!=expected_conditions:
                raise AssertionError("condition not registered")
            fs=[tuple(row["fingerprints"]) for row in saved]
            if baseline_fingerprints is None:baseline_fingerprints=fs
            elif baseline_fingerprints!=fs:
                raise AssertionError("different evaluation stream")
            if set(meta["train_fingerprints"]) & {f for pair in fs for f in pair}:
                raise AssertionError("train/eval leakage")
            reconstructed=evaluate(model,parent,record["eval_seed"],arm,r["smoke"])
            if reconstructed!=saved:
                raise AssertionError(f"unreplayable predictions for {seed}/{arm}")
            if summarize(saved)!=record["summaries"][arm]:
                raise AssertionError("unlinked summary")
            if record["summaries"][arm]["uninformative_bit_accuracy"]!=.5:
                raise AssertionError("neutralized-bit control failed")
            if any(row["teacher_bit_leak_control_only"]!=arm.startswith("oracle")
                   for row in saved):
                raise AssertionError("teacher bit mislabeled")
            evaluated+=2*len(saved)
        if parameter_hash(parent)!=parent_sha:
            raise AssertionError("parent changed during evaluation")
    return {"status":"REPLAY_VALIDATED_DEVELOPMENT_ONLY",
            "checkpoints":len(seeds)*len(ARMS),
            "model_episode_decisions":evaluated,
            "warning":"Oracle-bit arms explicitly use gold bits; this replays evaluation only."}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--parent",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(validate(a.parent,a.output),indent=2))
