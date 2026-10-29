#!/usr/bin/env python3
"""LANGMEM083 strict original-source 3-seed small-readout FIT vs new DEV.

Every source byte label is genuine newly registered original PSF publisher
documentation. The 3 seeds share 2 unseen owners so they do NOT constitute
independent corpora or independent factual retrieval tests.
"""
from __future__ import annotations
import argparse,hashlib,json,math,statistics
from pathlib import Path

SEEDS=(1701,1702,1703)
AGES=(8,32,128,256)
ARMS=("correct","address_only","wrong_key","shuffled_read","fit_prior")
PINNED={
 "Doc/howto/regex.rst":"69fd30034444ac74f456b3e9c14e852b6f127f08c9620bf703915450457c729b",
 "Doc/howto/argparse.rst":"b961e5773d40f6643588c62a8cbb767faae24896ffb23e778123bc3a25e8d3a4"
}

def summarize(root:Path):
    files=list(root.rglob("EVAL_RECEIPT.json"))
    if len(files)!=3:raise ValueError("requires exactly 3 original frozen-model seed new DEV reports")
    runs={}
    for p in files:
        r=json.loads(p.read_text())
        seed=r.get("seed")
        if seed not in SEEDS or seed in runs:raise ValueError("nonregistered/duplicate original 078 checkpoint seed")
        if (r.get("study")!="LANGMEM083"
            or r.get("status")!="SOURCE_SHA_FROZEN_NEW_PUBLISHER_DEV_AFTER_PROBE_FIT"
            or r.get("original_model_parameters_frozen")!=499627
            or r.get("trainable_probe_parameters_per_arm")!=9120
            or r.get("deployed_parameters_per_probe_with_frozen_base")!=508747
            or r.get("trained_steps_each_arm")!=120
            or r.get("trained_batch_each_arm")!=128
            or r.get("all_three_probe_checkpoint_hashes_frozen_before_DEV") is not True
            or r.get("new_unseen_document_sha256")!=PINNED
            or r.get("new_unseen_owner_count")!=2
            or r.get("original_model_state_bytes_per_stream")!=4160
            or r.get("extra_probe_weight_bytes_FP32")!=36480
            or r.get("frozen_model_parameters_unchanged") is not True
            or r.get("gpu_minutes")!=0 or r.get("protected_TEST")!="NOT_ACCESSED"):
            raise ValueError("original prefit registered probe, doc SHA or budget diverged")
        for h in ("original_pretrained_078_weights_sha256","FIT_historical_byte_SHA256",
                  "FIT_sample_index_SHA256","new_unseen_dev_historical_label_SHA256"):
            v=r.get(h)
            if not isinstance(v,str) or len(v)!=64:
                raise ValueError("source/pretrained/actual historical byte hash missing")
        rows=r.get("rows")
        if not isinstance(rows,list) or len(rows)!=r.get("new_unseen_historical_label_count"):
            raise ValueError("unseen original publisher source outcomes missing")
        if len(rows)<900:
            raise ValueError("frozen two owner source age cohort too small")
        ids=[]
        for q in rows:
            if (q["owner"] not in PINNED or q["age"] not in AGES or
                not 128<=q["clock"]<640 or q["clock"]%4!=0 or
                q["historical_clock"]!=q["clock"]-q["age"] or
                not 4<=q["historical_clock"]<q["clock"] or
                not 0<=q["historical_target_byte"]<=255 or
                q.get("source_role")!="UNSEEN_DEV"):
                raise ValueError("causal newly observed original source label/index mismatch")
            ids.append((q["owner"],q["clock"],q["age"],q["historical_clock"],q["historical_target_byte"]))
            for arm in ARMS:
                if (q[arm+"_correct"] not in (0,1)
                    or not math.isfinite(q[arm+"_nll"]) or q[arm+"_nll"]<0):
                    raise ValueError("invalid original historical byte probability prediction")
        if len(set((o,t,age) for o,t,age,_,_ in ids))!=len(ids):
            raise ValueError("duplicate actual publisher DEV historical query")
        if set(r.get("metrics",{}))!=set(map(str,AGES)):
            raise ValueError("source age results missing")
        for age in AGES:
            g=r["metrics"][str(age)]
            if g["count"]!=sum(q["age"]==age for q in rows):
                raise ValueError("real next-byte source age cohort changed")
        runs[seed]=r
    if set(runs)!=set(SEEDS):raise ValueError("3 original trained checkpoint seeds not all present")
    source_hashes={r["new_unseen_dev_historical_label_SHA256"] for r in runs.values()}
    if len(source_hashes)!=1:raise ValueError("new frozen source true historical bytes differ")
    ref=None
    for seed in SEEDS:
        ids=tuple((x["owner"],x["clock"],x["age"],x["historical_target_byte"]) for x in runs[seed]["rows"])
        if ref is None:ref=ids
        if ids!=ref:raise ValueError("original two new publishers ground truth differs across seed models")
    metrics={}
    for age in AGES:
        group=[q for r in runs.values() for q in r["rows"] if q["age"]==age]
        if len(group)<100:raise ValueError("minimum new source age group not achieved")
        metrics[str(age)]={"count_repeated_over_seeds":len(group),
            "unique_new_source_historical_targets_per_seed":len(group)//3}
        for arm in ARMS:
            metrics[str(age)][arm]={
                 "accuracy":statistics.mean(q[arm+"_correct"] for q in group),
                 "nll":statistics.mean(q[arm+"_nll"] for q in group)}
        metrics[str(age)]["correct_minus_address_only_nll"]=(
            metrics[str(age)]["correct"]["nll"]-
            metrics[str(age)]["address_only"]["nll"])
        metrics[str(age)]["correct_minus_wrong_key_nll"]=(
            metrics[str(age)]["correct"]["nll"]-
            metrics[str(age)]["wrong_key"]["nll"])
        metrics[str(age)]["correct_minus_fit_prior_nll"]=(
            metrics[str(age)]["correct"]["nll"]-
            metrics[str(age)]["fit_prior"]["nll"])
        metrics[str(age)]["correct_minus_address_only_accuracy"]=(
            metrics[str(age)]["correct"]["accuracy"]-
            metrics[str(age)]["address_only"]["accuracy"])
    return {"study":"LANGMEM083",
      "status":"FIRST_FROZEN_ORIGINAL_SLOW_MEMORY_TRAINED_READOUT_NEW_LICENSED_UNSEEN_DEV",
      "new_unseen_same_publisher_dev_owners":2,
      "full_original_078_model_seeds":3,
      "source_historical_label_SHA256":next(iter(source_hashes)),
      "unique_new_publisher_source_historical_labels_per_seed":len(ref),
      "reused_label_age_cells_across_three_seed_models":3*len(ref),
      "source_new_publisher_documents_SHA256":PINNED,
      "trained_auxiliary_readout_parameters_each":9120,
      "deployed_original_plus_single_probe_parameters":508747,
      "probe_fitted_comparator_arms":["correct","address_only","wrong_key"],
      "matched_optimizer_steps":120,"matched_sample_batch":128,
      "evaluation_done_after_checkpoints_SHA_sealed":True,
      "unseen_DEV_was_outside_original_FIT_and_exposed_DEV":True,
      "new_documents_same_PSF_publisher_domain":True,
      "metrics":metrics,
      "original_checkpoint_model_sha_by_seed":{str(s):runs[s]["original_pretrained_078_weights_sha256"] for s in SEEDS},
      "independently_qualified":False,
      "model_promoted":False,"gpu_minutes":0,
      "protected_TEST":"NOT_ACCESSED",
      "limits":["New author source documents from SAME original PSF CPython publisher; not an independent domain",
        "Old address externally selected with off-model history; no autonomous historical search or edge memory proof",
        "Accuracy under correlated source clock/age cells over only 2 source documents, model seed differences not independent doc samples",
        "Aux decoder 9120p trained but original 499627 source weights frozen; probe evaluation is NOT original next-byte NLL",
        "Snapshot of original slow state includes interference, no isolated event contribution features",
        "No protected TEST access; source whole graph IEEE/AD Lean certification OPEN"]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("new original source result must not overwrite")
    r=summarize(a.results)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LANGMEM083_NEW_DEV_AGGREGATE "+json.dumps({
      "unique_historical_targets":r["unique_new_publisher_source_historical_labels_per_seed"],
      "per_age":{age:{
        "n":m["unique_new_source_historical_targets_per_seed"],
        "correct":m["correct"],"address_only":m["address_only"],
        "wrong_key":m["wrong_key"],"fit_prior":m["fit_prior"],
        "correct_minus_address_nll":m["correct_minus_address_only_nll"]}
        for age,m in r["metrics"].items()}
    },sort_keys=True),flush=True)
if __name__=="__main__":main()
