"""Replay DEV-010 evaluations from original saved weights (not retraining).

Fail closed on changed source files, checkpoints, missing seed/cell, changed
counterfactual predictions and train/eval fingerprint overlap. This does not
certify untouched scientific holdouts or reproduction of SGD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from tac_osm.development.protected_memory import ProtectionModel  # noqa: E402
from tac_osm.development.finite_ced_memory import FiniteMemoryCEDModel  # noqa: E402
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402
from run_finite_ced_dev import (  # noqa: E402
    TRAIN_SEEDS,EVAL_SEEDS,eval_conditions,evaluate,summarize,
)


def validate(out: Path):
    report=json.loads((out/"report.json").read_text())
    if (report.get("study")!="DEV-010" or
            report.get("status")!="DEVELOPMENT_ONLY_NOT_CONFIRMATORY"):
        raise AssertionError("wrong experiment identity")
    for path,sha in report["source_hashes"].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:
            raise AssertionError("modified source: "+path)
    expected=TRAIN_SEEDS[:1] if report["smoke"] else TRAIN_SEEDS
    if len(report["records"])!=len(expected) or report["seeds"]!=list(expected):
        raise AssertionError("incomplete seed set")
    episodes=0
    for i,record in enumerate(report["records"]):
        if record["seed"]!=expected[i] or record["eval_seed"]!=EVAL_SEEDS[i]:
            raise AssertionError("mismatched seed pair")
        if set(record["arms"])!={"native","ced"}:
            raise AssertionError("model arms missing")
        training=record["training"]
        if training["total_train_examples"] != 4*report["steps"]:
            raise AssertionError("training example cardinality invalid")
        models={}
        for name,cls in (("native",ProtectionModel),("ced",FiniteMemoryCEDModel)):
            arm=record["arms"][name]
            filename=arm["file"]
            if Path(filename).name!=filename or not (out/filename).is_file():
                raise AssertionError("bad checkpoint file")
            if hashlib.sha256((out/filename).read_bytes()).hexdigest()!=arm["sha256"]:
                raise AssertionError("checkpoint changed")
            model=(cls(mode="always_update") if name=="native" else cls())
            model.load_state_dict(torch.load(out/filename,map_location="cpu",weights_only=True))
            if parameter_hash(model)!=training["final_hashes"][name]:
                raise AssertionError("checkpoint model weights inconsistent")
            if not training["writer_gradients_present"][name]:
                raise AssertionError("writer was untrained")
            model.eval()
            models[name]=model
        if set(record["rows"])!={"native","ced_dense","ced_sparse"}:
            raise AssertionError("evaluation mode missing")
        trainfps=set(training["train_visible_fingerprints"])
        if len(trainfps)!=4*report["steps"]:
            raise AssertionError("duplicate training inputs")
        for mode in ("native","ced_dense","ced_sparse"):
            saved=record["rows"][mode]
            if len(saved)!=len(eval_conditions(report["smoke"])):
                raise AssertionError("missing evaluator condition")
            evalfps={fp for row in saved for fp in row["fingerprints"]}
            if trainfps & evalfps:
                raise AssertionError("train/eval overlap")
            model=models["native" if mode=="native" else "ced"]
            reproduced=evaluate(model,record["eval_seed"],mode,report["smoke"])
            if reproduced!=saved:
                raise AssertionError("saved outcome not replayable: "+mode)
            if summarize(saved)!=record["summaries"][mode]:
                raise AssertionError("summary does not match episodes")
            if any(row["finite_memory_scalars_per_episode"]!=768 for row in saved):
                raise AssertionError("hidden memory capacity change")
            episodes+=sum(len(row["target"]) for row in saved)
        if any(a["fingerprints"]!=b["fingerprints"] for a,b in
               zip(record["rows"]["ced_dense"],record["rows"]["ced_sparse"])):
            raise AssertionError("dense/sparse evaluated different episodes")
    return {"status":"REPLAY_VALIDATED_DEVELOPMENT_ONLY",
            "checkpoints":2*len(report["records"]),
            "evaluated_model_episodes":episodes,
            "warning":"Replay validates evaluation from checkpoints; it does not reproduce training or show downstream capability."}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    args=p.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(validate(args.output),indent=2))
