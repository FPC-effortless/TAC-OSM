#!/usr/bin/env python3
"""ME001B independent recorded-only audit. NO refitting and NO publisher TEST.
Requires artifact 12225908024 extracted at root/results/me001b.
Uses original source preprocessing and predicts with the original trained weights.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
from scripts import seq010_real_recurrent as seq
from scripts.me001b_real_native_tac import (
    SEEDS, SPLIT_SALT, FEATURES, make_model, evaluate, train_thresholds,
    evaluate_zero_baseline,
)
MEMBER_SHA="e01cb0bd87dfab4c9dbad215d51d81282fc0d413be96d6f819ea872fb7a3c715"

def numeric_close(a,b,path,tol=2e-6):
    if a is None or b is None:
        if a is not b:raise AssertionError(f"{path}: missing mismatch {a} {b}")
    elif isinstance(a,(int,float)) and isinstance(b,(int,float)):
        if not math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol):
            raise AssertionError(f"{path}: {a} != {b}")
    elif a!=b:
        raise AssertionError(f"{path}: {a} != {b}")

def compare_metrics(a,b,label):
    if (a is None)!=(b is None): raise AssertionError(f"{label}: branch mismatch")
    if a is None:return
    if set(a)!=set(b):raise AssertionError(f"{label}: keys changed")
    for k in a:
        numeric_close(a[k],b[k],f"{label}.{k}")

def majority_signs(train,val,thresholds):
    # Training-only sign prior, no validation labels used to select direction.
    def select_signs(data):
        delta,mask=data[1:3]
        changed=(delta.abs()>thresholds[None,:]) & mask
        positive=((delta>0)&changed).sum(0).cpu().numpy()
        negative=((delta<0)&changed).sum(0).cpu().numpy()
        return positive,negative
    train_pos,train_neg=select_signs(train)
    val_pos,val_neg=select_signs(val)
    dominant=np.where(train_pos>=train_neg,1,-1)
    correct=np.where(dominant>0,val_pos,val_neg).sum()
    total=(val_pos+val_neg).sum()
    return {
        "train_positive_per_feature":train_pos.tolist(),
        "train_negative_per_feature":train_neg.tolist(),
        "validation_positive_total":int(val_pos.sum()),
        "validation_negative_total":int(val_neg.sum()),
        "train_majority_sign_accuracy_on_heldout_changes":float(correct/total),
        "global_positive_sign_accuracy_on_heldout_changes":float(val_pos.sum()/total),
        "global_negative_sign_accuracy_on_heldout_changes":float(val_neg.sum()/total),
        "train_only_majority_sign":dominant.tolist(),
        "change_count":int(total),
    }

def audit(artifact_root,output):
    torch.set_num_threads(1)
    artifact_root=Path(artifact_root)
    recorded=json.loads((artifact_root/"results.json").read_text())
    source=json.loads((artifact_root/"source.json").read_text())
    assert recorded["study"]=="ME001B"
    assert not recorded["official_validation_read"] and not recorded["official_test_read"]
    assert not source["publisher_validation_entries_read"] and not source["test_entries_read"]
    assert source["train_member_sha256"]==MEMBER_SHA
    assert recorded["source"]["train_member_sha256"]==MEMBER_SHA
    assert recorded["split_salt"]==SPLIT_SALT
    assert len(recorded["fits"])==9
    hashes={}
    for e in recorded["fits"]:
        f=artifact_root/e["checkpoint"]
        digest=hashlib.sha256(f.read_bytes()).hexdigest()
        if digest!=e["sha256"]:raise ValueError("checkpoint hash mismatch "+str(f))
        hashes[e["model"]+"-"+str(e["seed"])]=digest

    # Authentic TRAIN-only data; independent download, model weights frozen.
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    seq.SPLIT_SALT=SPLIT_SALT
    file,replay_source=seq.acquire_training_csv(output)
    if replay_source["train_member_sha256"]!=MEMBER_SHA:
        raise ValueError("TRAIN member SHA changed: abort (ZIP may be repackaged)")
    tr,va,cohort=seq.load_real_windows(file)
    file.unlink(missing_ok=True)
    train,val=seq.prepare(*tr),seq.prepare(*va)
    thresholds=train_thresholds(train)
    for i,(x,y) in enumerate(zip(thresholds.tolist(),recorded["thresholds_training_only"])):
        numeric_close(x,y,f"threshold[{i}]",tol=1e-6)
    if cohort != recorded["cohort"]:raise AssertionError("source split cohort differs")
    compare_metrics(seq.persistence_score(val),recorded["baseline"]["persistence"],"persistence")
    compare_metrics(evaluate_zero_baseline(val,thresholds),recorded["baseline"]["change"],"persistence changes")
    replay=[]
    for e in recorded["fits"]:
        torch.manual_seed(e["seed"])
        model=make_model(e["model"])
        ck=torch.load(artifact_root/e["checkpoint"],weights_only=True,map_location="cpu")
        assert ck["model"]==e["model"] and ck["seed"]==e["seed"]
        assert ck["steps"]==e["steps"]
        model.load_state_dict(ck["state_dict"],strict=True)
        got=evaluate(model,val,thresholds=thresholds)
        compare_metrics(got,e["validation"],str(e["model"])+str(e["seed"]))
        reset=evaluate(model,val,thresholds=thresholds,reset_at=seq.HISTORY//2) if e["model"]=="native_tac_assembly" else None
        compare_metrics(reset,e["reset_midway_control"],"reset "+str(e["seed"]))
        replay.append({"model":e["model"],"seed":e["seed"],"macro_mae":got["macro_mae"],
                       "change_macro_mae":got["change_macro_mae"],
                       "replay_matches_recorded":True})
    sign=majority_signs(train,val,thresholds)
    assert sign["change_count"]==recorded["baseline"]["change"]["change_total"]
    results={"study":"ME001B recorded-only source and checkpoint replay",
             "replay_confirmed":True,"fits_replayed":len(replay),
             "refits":0,"official_publisher_test_used":False,
             "official_publisher_validation_used":False,
             "source_member_sha256":MEMBER_SHA,
             "original_source_zip_sha256":source["zip_sha256"],
             "replay_source_zip_sha256":replay_source["zip_sha256"],
             "cohort":cohort,"checkpoint_sha256":hashes,
             "replay_scores":replay,"train_only_change_sign_baseline":sign,
             "architecture_promoted":False}
    (output/"audit.json").write_text(json.dumps(results,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"PASS","fits":len(replay),"refits":0,
                      "train_sign_baseline":sign["train_majority_sign_accuracy_on_heldout_changes"]},sort_keys=True))
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--artifact-root",default="results/me001b")
    p.add_argument("--output",default="results/me001b-audit")
    a=p.parse_args()
    audit(a.artifact_root,a.output)
if __name__=="__main__":
    main()
