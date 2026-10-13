"""DEV-011 fail-closed checkpoint + frozen-state evaluation replay.

Recomputes historical states from immutable DEV-010 parent writers and
regenerates every held-out example/decision using saved probe weights.
Does *not* reproduce probe optimization; checkpoint replay is not scientific
confirmation or evidence that no information exists in a failed probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from run_frozen_plm_probes_dev import (
    PARENT_SHA256, PROBES, SEEDS, WRITERS, DecoderProbe,
    prepare_parent, open_writer, encode_examples, training_cases,
    evaluation_cases, evaluate_probe, EVAL_SEEDS, ROOT,
)


def validate(parent_archive: Path, output: Path):
    parent=prepare_parent(parent_archive)
    report=json.loads((output/"report.json").read_text())
    if report.get("study")!="DEV-011" or report.get("status")!="DEVELOPMENT_ONLY_NOT_CONFIRMATORY":
        raise AssertionError("invalid study/status")
    if report.get("parent_artifact_sha256")!=PARENT_SHA256:
        raise AssertionError("wrong parent hash")
    if report.get("origins")!=list(WRITERS) or report.get("probe_types")!=list(PROBES):
        raise AssertionError("missing experimental arm")
    expected=SEEDS[:1] if report.get("smoke") else SEEDS
    if report.get("seeds")!=list(expected) or len(report.get("records",[]))!=len(expected):
        raise AssertionError("missing seed")
    for path, sha in report["source_sha256"].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:
            raise AssertionError("source changed since experiment")
    total=0
    probes=0
    for rec,seed in zip(report["records"], expected):
        if rec["seed"]!=seed or set(rec["origins"])!=set(WRITERS):
            raise AssertionError("wrong seed or incomplete origin")
        origin_eval_rows=None
        for origin in WRITERS:
            saved=rec["origins"][origin]
            writer=open_writer(parent_archive,parent,seed,origin)
            from tac_osm.development.noisy_memory import parameter_hash
            frozen_sha=parameter_hash(writer)
            if saved["frozen_writer_parameter_hash"]!=frozen_sha:
                raise AssertionError("writer weights mismatch")
            train_rows, _=encode_examples(writer,training_cases(seed,report["smoke"]))
            eval_rows,features=encode_examples(writer,evaluation_cases(
                EVAL_SEEDS[SEEDS.index(seed)],report["smoke"]))
            if saved["train_rows"]!=train_rows or saved["eval_rows"]!=eval_rows:
                raise AssertionError("changed frozen observations/provenance")
            train_fps={fp for row in train_rows for fp in row["fingerprints"]}
            test_fps={fp for row in eval_rows for fp in row["fingerprints"]}
            if train_fps & test_fps or len(train_fps)!=2*len(train_rows):
                raise AssertionError("invalid train/eval split")
            if origin_eval_rows is None:
                origin_eval_rows=eval_rows
            elif origin_eval_rows != eval_rows:
                # state_pair_l2 is writer-dependent, so compare only public data.
                allowed=("partition","index","h","d","near","noise",
                         "role","fingerprints","target","oracle_predictions","oracle_inspected")
                for first,second in zip(origin_eval_rows,eval_rows):
                    if any(first[k]!=second[k] for k in allowed):
                        raise AssertionError("different public eval data by origin")
            if set(saved["probes"])!=set(PROBES):
                raise AssertionError("probe arm missing")
            for kind in PROBES:
                arm=saved["probes"][kind]
                name=arm["file"]
                if (Path(name).name!=name or name!=f"{origin}-{kind}-seed{seed}.pt"):
                    raise AssertionError("invalid checkpoint name")
                f=output/name
                if hashlib.sha256(f.read_bytes()).hexdigest()!=arm["sha256"]:
                    raise AssertionError("modified probe checkpoint")
                model=DecoderProbe(kind)
                model.load_state_dict(torch.load(f,map_location="cpu",weights_only=True),strict=True)
                model.eval()
                if sum(p.numel() for p in model.parameters())!=arm["parameters"]:
                    raise AssertionError("parameter count inconsistent")
                replay=evaluate_probe(model,features,eval_rows)
                reset=evaluate_probe(model,features,eval_rows,zero_memory=True)
                if replay!=arm["heldout"] or reset["accuracy"]!=arm["reset_accuracy"]:
                    raise AssertionError(f"unreplayable predictions {seed} {origin} {kind}")
                total+=replay["episodes"]
                probes+=1
            if parameter_hash(writer)!=frozen_sha:
                raise AssertionError("writer changed after evaluation")
    return {
        "status":"REPLAY_VALIDATED_DEVELOPMENT_ONLY",
        "parent_checkpoints":2*len(expected),
        "frozen_writer_origins":len(expected)*len(WRITERS),
        "probe_checkpoints":probes,
        "probe_model_episode_evaluations":total,
        "warning":"This replays evaluation, not probe SGD optimization or causal memory sufficiency."
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--parent",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(validate(a.parent,a.output),indent=2))
