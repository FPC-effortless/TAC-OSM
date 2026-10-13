"""DEV-009 saved-checkpoint evaluation replay. Does not rerun training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from tac_osm.development.ced_hardened_router import QueryOnlyCED  # noqa: E402
from run_ced_hardened_dev import _hash_weights, evaluate_loop  # noqa: E402
from run_ced_index_gradient_dev import measure_index_recall  # noqa: E402


def validate(folder: Path) -> dict:
    r=json.loads((folder/"report.json").read_text())
    if r["study"]!="DEV-009" or r["classification"]!="DEVELOPMENT_ONLY_NOT_CONFIRMATORY":
        raise AssertionError("invalid experiment identity")
    for path, digest in r["source_sha256"].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
            raise AssertionError("modified source: "+path)
    if not r["records"]:
        raise AssertionError("empty report")
    checked=0
    recall_cases=0
    for item in r["records"]:
        arms=item["arms"]
        if set(arms)!={"shared","sparse"} or arms["shared"]["initial_hash"]!=arms["sparse"]["initial_hash"]:
            raise AssertionError("wrong paired arms or mismatched initialization")
        for mode,entry in arms.items():
            path=folder/entry["checkpoint"]
            if path.name!=entry["checkpoint"] or hashlib.sha256(path.read_bytes()).hexdigest()!=entry["checkpoint_sha256"]:
                raise AssertionError("modified or missing checkpoint")
            model=QueryOnlyCED(mode=mode,topk=4)
            model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
            model.eval()
            if _hash_weights(model)!=entry["final_hash"] or _hash_weights(model)==entry["initial_hash"]:
                raise AssertionError("checkpoint parameter mismatch")
            if len(entry["cells"])!=4 or {
                    (x["candidates"],x["delay"]) for x in entry["cells"]}!={(8,0),(8,4),(32,0),(32,4)}:
                raise AssertionError("missing evaluation cells")
            for cell in entry["cells"]:
                if len(cell["raw"])!=r["episodes_per_cell"]:
                    raise AssertionError("missing raw episode")
                reproduced=evaluate_loop(model,seed=item["seed"]+987,
                    n=cell["candidates"],delay=cell["delay"],
                    samples=r["episodes_per_cell"])
                if reproduced!=cell:
                    raise AssertionError("evaluation replay differs")
                checked+=len(cell["raw"])
            if len(entry["index_recall"])!=2:
                raise AssertionError("missing index recalls")
            for x in entry["index_recall"]:
                reproduced=measure_index_recall(model,item["seed"]+1234,x["candidates"],x["trials"])
                if x!=reproduced:
                    raise AssertionError("index recall replay differs")
                recall_cases+=x["trials"]
    return {"status":"REPLAY_VALIDATED_DEVELOPMENT_ONLY",
            "checkpoints":len(r["records"])*2,
            "task_episodes":checked,"index_recall_cases":recall_cases,
            "warning":"This validates evaluation from saved checkpoints, not training."}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(validate(args.folder),indent=2))
