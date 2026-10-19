#!/usr/bin/env python3
"""SEQ010 recorded-only independent checkout: no training, no publisher TEST.

Restores original checkpoint artifacts and repeats identical real-world
TRAIN-internal scored predictions against original archived metrics.
This is numerical evidence verification, not external independent physics.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seq010_real_recurrent import (
    MODELS, SEEDS, ResidualRecurrent, acquire_training_csv,
    load_real_windows, prepare, persistence_score, score
)

SOURCE_SHA = "46ca1cc9d691a31bead8a58c8710b8c83201bb0d"
ORIGINAL_RUN = 39317978344

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git_blob(path):
    data=Path(path).read_bytes()
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()

def audit(original, output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    repo_root=Path(__file__).resolve().parent.parent
    if git_blob(repo_root/"scripts"/"seq010_real_recurrent.py") != SOURCE_SHA:
        raise ValueError("Source changed from original recorded fit")
    report=json.loads((original/"results.json").read_text())
    recorded=json.loads((original/"source.json").read_text())
    assert report["study"]=="SEQ010"
    assert report["stage"]=="TRAIN_INTERNAL_EXPLORATORY_REAL"
    assert not report["official_publisher_validation_used"]
    assert not report["official_publisher_test_used"]
    assert not recorded["test_entries_read"]
    assert not recorded["publisher_validation_entries_read"]
    assert report["source"]==recorded
    assert len(report["recurrent"])==4
    observed={(r["kind"],r["seed"]) for r in report["recurrent"]}
    expected={(k,s) for k in MODELS for s in SEEDS}
    if observed!=expected:raise ValueError("Original incomplete/mismatched fit matrix")

    # Redownload the identical real source, then reconstruct original
    # fixed split, real inputs, missingness masks and exact heldout.
    output.mkdir(parents=True,exist_ok=True)
    source_dir=output/"source"
    source_dir.mkdir(exist_ok=True)
    archive=source_dir/"official-v3.zip"
    try:
        source_csv,source=acquire_training_csv(source_dir)
    finally:
        archive.unlink(missing_ok=True)
    for field in ("zip_sha256","train_member_sha256","train_member_bytes",
                  "publisher_zip_url","dataset_doi","dataset_version"):
        if source[field]!=recorded[field]:
            raise ValueError("Actual source mismatch "+field)
    train,validation,cohort=load_real_windows(source_csv)
    original_cohort=report["cohort"]
    if cohort!=original_cohort:
        raise ValueError("Vehicle/sensor cohort changed")
    inputs=prepare(*validation)
    baseline=persistence_score(inputs)
    original_baseline=report["baseline"]["persistence"]
    for key,val in baseline.items():
        if abs(val-original_baseline[key])>1e-10:
            raise ValueError(f"Persistence replay drift {key}: {val}")
    rows=[]
    for row in report["recurrent"]:
        p=original/row["weights"]
        if sha(p)!=row["checkpoint_sha256"]:
            raise ValueError("Original checkpoint SHA mismatched")
        checkpoint=torch.load(p,map_location="cpu",weights_only=True)
        if checkpoint["kind"]!=row["kind"] or checkpoint["seed"]!=row["seed"]:
            raise ValueError("Original checkpoint identity mismatch")
        model=ResidualRecurrent(row["kind"])
        model.load_state_dict(checkpoint["weights"], strict=True)
        measured=score(model,inputs)
        for key,value in measured.items():
            if abs(value-row["heldout"][key])>1e-8:
                raise ValueError(f"Score drift {row['kind']}/{row['seed']}/{key}")
        rows.append({"kind":row["kind"],"seed":row["seed"],
                     "sha256":row["checkpoint_sha256"],
                     "measured":measured,"exact_match":True})
    result={"status":"PASS_SAME_SOURCE_RECORDED_REPLAY",
            "original_training_run":ORIGINAL_RUN,
            "original_checkpoints_verified":len(rows),
            "new_model_fits":0,"published_test_opened":False,
            "published_validation_opened":False,
            "source_match":True,"cohort_match":True,
            "baseline":baseline,"replayed":rows}
    (output/"audit.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"verified":len(rows),
                      "new_fits":0,"test_opened":False}),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--original",required=True)
    parser.add_argument("--output",default="results/seq010-replay")
    args=parser.parse_args()
    audit(Path(args.original),Path(args.output))

if __name__=="__main__":
    main()
