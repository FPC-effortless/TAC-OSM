"""DEV-013 source/checkpoint integrity and complete held-out evaluator replay.

This validates predictions from saved trained weights. It does NOT reproduce
training, establish independent scientific confirmation or parameter/FLOP
equivalence beyond this controlled model family.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from tac_osm.development.noisy_memory import parameter_hash
from tac_osm.development.casm_readout_alignment import ARMS,CASMAlignedModel
from run_casm_alignment_dev import (ROOT,SEEDS,EVAL_SEEDS,conditions,evaluate,summarize)


def validate(folder: Path) -> dict:
    r=json.loads((folder/"report.json").read_text())
    if r.get("study")!="DEV-013" or r.get("status")!="DEVELOPMENT_ONLY_NOT_CONFIRMATORY":
        raise AssertionError("wrong experiment classification")
    seeds=SEEDS[:1] if r.get("smoke") else SEEDS
    if r.get("seeds")!=list(seeds) or r.get("arms")!=list(ARMS):
        raise AssertionError("missing seeds/arms")
    if len(r.get("records",[]))!=len(seeds):
        raise AssertionError("missing seed record")
    for path,digest in r["source_sha256"].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
            raise AssertionError("source mutated: "+path)
    model_episodes=0
    for seed,record in zip(seeds,r["records"]):
        if seed!=record["seed"] or record["eval_seed"]!=EVAL_SEEDS[SEEDS.index(seed)]:
            raise AssertionError("wrong evaluator seed mapping")
        meta=record["training"]
        if set(meta["initial_hashes"])!=set(ARMS) or len(set(meta["initial_hashes"].values()))!=1:
            raise AssertionError("unmatched starting weights")
        if not all(meta["writer_gradient"].values()) or not meta["casm_aux_gradient"]["aligned_aux"]:
            raise AssertionError("writer/CASM gradient evidence missing")
        training=set(meta["train_fingerprints"])
        if len(training)!=r["steps"]*4:
            raise AssertionError("duplicate training events")
        if set(record["models"])!=set(ARMS) or set(record["rows"])!=set(ARMS):
            raise AssertionError("incomplete arms")
        fp0=None
        for arm in ARMS:
            saved=record["models"][arm]
            f=saved["file"]
            if f!=f"{arm}-seed{seed}.pt" or Path(f).name!=f:
                raise AssertionError("unsafe or wrong checkpoint name")
            path=folder/f
            if hashlib.sha256(path.read_bytes()).hexdigest()!=saved["checkpoint_sha256"]:
                raise AssertionError("checkpoint bytes differ")
            m=CASMAlignedModel()
            m.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True)
            m.eval()
            if parameter_hash(m)!=meta["trained_hashes"][arm]:
                raise AssertionError("checkpoint tensor digest differs")
            if parameter_hash(m)==meta["initial_hashes"][arm]:
                raise AssertionError("model untrained")
            if sum(p.numel() for p in m.parameters())!=saved["parameter_count"]:
                raise AssertionError("parameter count differs")
            rows=record["rows"][arm]
            if len(rows)!=len(conditions(r["smoke"])):
                raise AssertionError("missing evaluator cell")
            ordered=[tuple(sorted(row["condition"].items())) for row in rows]
            registered=[tuple(sorted({"h":h,"d":d,"near":near,"noise":noise,
                    "role":role,"rep":rep}.items()))
                    for h,d,near,noise,role,rep in conditions(r["smoke"])]
            if ordered!=registered:
                raise AssertionError("condition order modified")
            efp=[tuple(row["fingerprints"]) for row in rows]
            if fp0 is None:fp0=efp
            elif fp0!=efp:raise AssertionError("different evaluation examples by arm")
            if training & {fp for pair in efp for fp in pair}:
                raise AssertionError("training/eval overlap")
            regenerated=evaluate(m,record["eval_seed"],arm,r["smoke"])
            if regenerated!=rows:
                raise AssertionError(f"unreplayable predictions {seed}/{arm}")
            if summarize(rows)!=record["summaries"][arm]:
                raise AssertionError("summary not linked to raw observations")
            if record["summaries"][arm]["reset_task_accuracy"]!=.5 or (
                    record["summaries"][arm]["reset_bit_accuracy"]!=.5):
                raise AssertionError("zero-state leakage")
            if any(row["state_scalars_per_episode"]!=768 for row in rows):
                raise AssertionError("PLM state capacity changed")
            model_episodes+=2*len(rows)
    return {"status":"REPLAY_VALIDATED_DEVELOPMENT_ONLY",
            "checkpoints":len(seeds)*len(ARMS),
            "model_episode_predictions":model_episodes,
            "warning":"Evaluation replay does not reproduce training optimization or prove meaningful world modeling."}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("folder",type=Path)
    a=p.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(validate(a.folder),indent=2))
