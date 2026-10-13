"""DEV-012 checkpoint evaluation replay: never mistake this for training replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from run_rehearsal_writer_dev import (
    ARMS, EVAL_SEEDS, ROOT, SEEDS, conditions, evaluate, summarize,
)
from tac_osm.development.noisy_memory import parameter_hash
from tac_osm.development.rehearsal_writer import RehearsalWriterModel


def validate(out: Path):
    report=json.loads((out/"report.json").read_text())
    if report.get("study")!="DEV-012" or report.get("status")!="DEVELOPMENT_ONLY_NOT_CONFIRMATORY":
        raise AssertionError("wrong study identity or promotion claim")
    expected=SEEDS[:1] if report["smoke"] else SEEDS
    if report.get("seeds")!=list(expected) or report.get("arms")!=list(ARMS):
        raise AssertionError("missing seed or treatment arm")
    if len(report.get("records",[]))!=len(expected):
        raise AssertionError("incomplete run")
    for path, sha in report["source_sha256"].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:
            raise AssertionError("source changed: "+path)
    rows_checked=0
    for i,record in enumerate(report["records"]):
        seed=expected[i]
        if record["seed"]!=seed or record["eval_seed"]!=EVAL_SEEDS[i]:
            raise AssertionError("wrong training/evaluation seed pairing")
        if set(record["rows"])!=set(ARMS) or set(record["models"])!=set(ARMS):
            raise AssertionError("incomplete arm rows")
        meta=record["training"]
        if set(meta["initial_hashes"])!=set(ARMS) or len(set(meta["initial_hashes"].values()))!=1:
            raise AssertionError("starting weights not identical")
        if meta["samples_per_arm"]!=4*report["steps"]:
            raise AssertionError("incorrect training size")
        if not all(meta["writer_gradients"].values()) or not meta["learned_protection_gradient"]:
            raise AssertionError("missing writer/gate gradient evidence")
        trainfps=set(meta["training_fingerprints"])
        if len(trainfps)!=4*report["steps"]:
            raise AssertionError("training streams duplicated")
        baseline_fps=None
        for arm in ARMS:
            info=record["models"][arm]
            filename=info["file"]
            if filename!=f"{arm}-seed{seed}.pt" or Path(filename).name!=filename:
                raise AssertionError("invalid checkpoint path")
            if hashlib.sha256((out/filename).read_bytes()).hexdigest()!=info["sha256"]:
                raise AssertionError("checkpoint mismatch")
            model=RehearsalWriterModel(mode=info["mode"])
            model.load_state_dict(torch.load(out/filename,map_location="cpu",weights_only=True))
            model.eval()
            if parameter_hash(model)!=meta["trained_hashes"][arm]:
                raise AssertionError("tensor hash mismatch")
            if parameter_hash(model)==meta["initial_hashes"][arm]:
                raise AssertionError("not trained")
            if sum(p.numel() for p in model.parameters() if p.requires_grad)!=info["trainable_parameters"]:
                raise AssertionError("trainable-parameter count mismatch")
            saved=record["rows"][arm]
            if len(saved)!=len(conditions(report["smoke"])):
                raise AssertionError("missing evaluation condition")
            keys=[tuple(sorted(row["condition"].items())) for row in saved]
            expected_conditions=[tuple(sorted({"h":h,"d":d,"near":near,"noise":noise,"role":role,
                                               "rep":rep}.items()))
                                 for h,d,near,noise,role,rep in conditions(report["smoke"])]
            if keys!=expected_conditions:
                raise AssertionError("missing/changed condition row")
            evaluation_fps={fp for row in saved for fp in row["fingerprints"]}
            if trainfps & evaluation_fps:
                raise AssertionError("train/eval contamination")
            ordered_fps=[tuple(row["fingerprints"]) for row in saved]
            if baseline_fps is None:
                baseline_fps=ordered_fps
            elif ordered_fps!=baseline_fps:
                raise AssertionError("evaluation stream differs across arms")
            reproduced=evaluate(model,record["eval_seed"],report["smoke"])
            if reproduced!=saved:
                raise AssertionError("unreplayable episode predictions: "+arm)
            if summarize(saved)!=record["summaries"][arm]:
                raise AssertionError("unreplayable metric summary: "+arm)
            if record["summaries"][arm]["reset_task_accuracy"]!=.5 or (
                    record["summaries"][arm]["reset_bit_accuracy"]!=.5):
                raise AssertionError("zero-state leakage")
            if any(row["memory_scalars_per_episode"]!=768 for row in saved):
                raise AssertionError("unreported state capacity change")
            rows_checked+=2*len(saved)
    return {
        "status":"REPLAY_VALIDATED_DEVELOPMENT_ONLY",
        "checkpoints":len(expected)*len(ARMS),
        "model_episode_predictions":rows_checked,
        "warning":"This regenerates held-out inference from weights, NOT training or independent scientific replication.",
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    args=p.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(validate(args.output),indent=2))
