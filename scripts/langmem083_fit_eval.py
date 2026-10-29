#!/usr/bin/env python3
"""LANGMEM083: frozen original FIT-only probe training and unseen DEV scoring.

train phase has NO unseen-dir argument; evaluate requires SHA-sealed trained
three heads, their optimizer/RNG receipts, and SHA-checked new publisher docs.
"""
from __future__ import annotations
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research/mathematical-engineering/langmem083"))
from frozen_probe import (
    ARMS,AGES,STEPS,BATCH,FEATURE_COUNT,DEPLOYED_PARAMETERS,
    SEEDS,FrozenReadout083,causal_feature_set,run_training,
    fit_sources,restore_original,source_tensor_digest)

def read_unseen_publisher_docs(unseen_dir):
    reg=json.loads((ROOT/"research/mathematical-engineering/langmem083/NEW_DEV_SOURCE.json").read_text())
    docs=reg["documents"]
    if reg["commit"]!="60403a5409ff2c3f3b07dd2ca91a7a3e096839c7" or len(docs)!=2:
        raise ValueError("publisher DEV source receipt differs")
    if set(x["path"] for x in docs)!={"Doc/howto/regex.rst","Doc/howto/argparse.rst"}:
        raise ValueError("wrong frozen DEV source paths")
    out={}
    for e in docs:
        path=unseen_dir/Path(e["path"]).name
        if not path.is_file():raise ValueError("missing frozen unseen DEV document")
        raw=path.read_bytes()
        git_blob=hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()
        if (len(raw)!=e["bytes"] or
            hashlib.sha256(raw).hexdigest()!=e["sha256"] or
            git_blob!=e["git_blob"]):
            raise ValueError("original publisher DEV SHA/blob/byte length mismatch")
        out[e["path"]]=raw
    if len(list(unseen_dir.glob("*.rst")))!=2:
        raise ValueError("unexpected unseen extra publisher DEV document")
    return out

@torch.no_grad()
def evaluate(seed,model,trained,unseen,out):
    if seed not in SEEDS or out.exists():
        raise ValueError("nonregistered seed or original output overwrite")
    frozen_hash=source_tensor_digest(model)
    rpath=trained/"FIT_RECEIPT.json"
    if not rpath.is_file():raise ValueError("unseen publisher DEV cannot precede trained frozen heads")
    fit=json.loads(rpath.read_text())
    if (fit.get("study")!="LANGMEM083" or fit.get("seed")!=seed or
        fit.get("num_original_fit_owners")!=14 or
        fit.get("trained_updates_per_arm")!=STEPS or
        fit.get("batch_size_per_arm")!=BATCH or
        fit.get("original_encoder_weights_sha256_before")!=frozen_hash or
        fit.get("original_encoder_weights_sha256_after")!=frozen_hash or
        fit.get("unseen_DEV_accessed_before_probes_frozen") is not False or
        set(fit.get("checkpoint_receipts",{}))!=set(ARMS)):
        raise ValueError("original FIT-only probe checkpoint receipt not sealed")
    heads={}
    for arm in ARMS:
        receipt=fit["checkpoint_receipts"][arm]
        file=trained/receipt["name"]
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=receipt["checkpoint_sha256"]:
            raise ValueError("source-only trained probe checkpoint SHA changed")
        item=torch.load(file,map_location="cpu",weights_only=True)
        if (item["study"]!="LANGMEM083" or item["arm"]!=arm or item["seed"]!=seed or
            item["steps"]!=STEPS or item["original_model_weight_sha256"]!=frozen_hash or
            item["source_fit_target_sha256"]!=fit["fit_historical_target_sha256"] or
            item["fit_sampler_sha256"]!=fit["fit_source_sampler_sha256"]):
            raise ValueError("saved original trained auxiliary decoder missing source SHA")
        head=FrozenReadout083().eval()
        head.load_state_dict(item["probe_model"],strict=True)
        heads[arm]=head
    # The first NEW unseen DEV disk read in this experiment is now: AFTER
    # all 3 auxiliary heads' trained weights/optimizer/RNG hashes sealed.
    docs=read_unseen_publisher_docs(unseen)
    begin=time.process_time()
    xs,y,meta,source_sha=causal_feature_set(model,docs,source_role="UNSEEN_DEV")
    if len(y)<100:raise ValueError("new authentic source historical target cohort too small")
    logits={arm:heads[arm](xs[arm]) for arm in ARMS}
    shifted=xs["correct"].clone()
    shifted[:,:8]=shifted[:,:8].roll(1,dims=0)
    logits["shuffled_read"]=heads["correct"](shifted)
    counts=torch.tensor(fit["fit_frequency_by_age"],dtype=torch.float64)
    if counts.shape!=(4,256) or not (counts.sum(dim=1)>0).all():
        raise ValueError("age frequency FIT source prior missing")
    priors=(counts+8/256)/(counts.sum(dim=1,keepdim=True)+8)
    rows=[]
    for i,(owner,t,a,age) in enumerate(meta):
        if age not in AGES or not 128<=t<640 or a!=t-age or a<4:
            raise ValueError("source observed historical clock incorrect")
        label=int(y[i]);idx=AGES.index(age)
        row={"owner":owner,"clock":t,"historical_clock":a,"age":age,
             "historical_target_byte":label,"source_role":"UNSEEN_DEV"}
        for arm in ("correct","address_only","wrong_key","shuffled_read"):
            lp=F.log_softmax(logits[arm][i].float(),dim=0)
            row[arm+"_nll"]=-float(lp[label])
            row[arm+"_correct"]=int(int(lp.argmax())==label)
        row["fit_prior_nll"]=-math.log(float(priors[idx,label]))
        row["fit_prior_correct"]=int(int(priors[idx].argmax())==label)
        rows.append(row)
    digest=hashlib.sha256()
    for r in rows:
        digest.update(r["owner"].encode()+b":"+str(r["clock"]).encode()+b":"+
                      str(r["historical_clock"]).encode()+b":"+
                      bytes([r["historical_target_byte"]])+b";")
    metrics={}
    for age in AGES:
        sample=[r for r in rows if r["age"]==age]
        if not sample:raise ValueError("missing genuine source age cohort")
        metrics[str(age)]={"count":len(sample)}
        for arm in ("correct","address_only","wrong_key","shuffled_read","fit_prior"):
            metrics[str(age)][arm]={
                "accuracy":sum(r[arm+"_correct"] for r in sample)/len(sample),
                "nll":sum(r[arm+"_nll"] for r in sample)/len(sample)}
    if source_tensor_digest(model)!=frozen_hash:
        raise ValueError("original 499627 full model changed during no-grad probe evaluation")
    return {
      "study":"LANGMEM083","seed":seed,
      "status":"SOURCE_SHA_FROZEN_NEW_PUBLISHER_DEV_AFTER_PROBE_FIT",
      "original_pretrained_078_weights_sha256":frozen_hash,
      "original_model_parameters_frozen":499627,
      "trainable_probe_parameters_per_arm":FEATURE_COUNT,
      "deployed_parameters_per_probe_with_frozen_base":DEPLOYED_PARAMETERS,
      "trained_steps_each_arm":STEPS,"trained_batch_each_arm":BATCH,
      "FIT_historical_byte_SHA256":fit["fit_historical_target_sha256"],
      "FIT_sample_index_SHA256":fit["fit_source_sampler_sha256"],
      "all_three_probe_checkpoint_hashes_frozen_before_DEV":True,
      "new_unseen_document_sha256":{k:hashlib.sha256(v).hexdigest() for k,v in docs.items()},
      "new_unseen_dev_historical_label_SHA256":digest.hexdigest(),
      "new_unseen_owner_count":len(docs),
      "new_unseen_historical_label_count":len(rows),
      "original_model_state_bytes_per_stream":4160,
      "extra_probe_weight_bytes_FP32":FEATURE_COUNT*4,
      "target_semantics":"HISTORICAL_OBSERVED_SOURCE_BYTE_AT_EVALUATOR_SUPPLIED_OLD_ADDRESS",
      "historical_keys_side_table":"EVALUATOR_ONLY_NOT_TACS_4160_BYTES",
      "metrics":metrics,"rows":rows,
      "cpu_dev_seconds":time.process_time()-begin,
      "gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
      "independently_qualified":False,
      "limitations":["New docs unseen during original FIT and probe FIT, but same CPython publisher/domain",
        "External correct old writer key and plaintext side-table used by probe, not autonomous TAC history retrieval",
        "Two new source owners, highly correlated observations across ages and clocks",
        "Probe requires 9120 new trained weights /36480B FP32, no memory payload guarantee for selector",
        "Shuffled read feature is a frozen correct-head distribution shift control",
        "This predicts historical source byte, NOT original next-byte model quality",
        "Whole original model source IEEE/AD proof and source-independent domain replication OPEN"]
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("train","evaluate"))
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--original-checkpoint-dir",type=Path,required=True)
    p.add_argument("--original-aggregate",type=Path,required=True)
    p.add_argument("--trained-dir",type=Path,required=True)
    p.add_argument("--unseen-dir",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    torch.set_num_threads(2)
    original,_=restore_original(a.seed,a.original_checkpoint_dir,a.original_aggregate)
    original.eval()
    for w in original.parameters():w.requires_grad_(False)
    if a.phase=="train":
        if a.unseen_dir is not None or a.output is not None:
            raise ValueError("FIT-only phase cannot receive any unseen DEV path")
        run_training(a.seed,original,fit_sources(),a.trained_dir)
    else:
        if a.unseen_dir is None or a.output is None:
            raise ValueError("unseen DEV phase requires registered paths")
        r=evaluate(a.seed,original,a.trained_dir,a.unseen_dir,a.output)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+"\n")
        print("LANGMEM083_NEW_UNSEEN_DEV "+json.dumps({
          "seed":a.seed,"original_labels":r["new_unseen_historical_label_count"],
          "source_SHA":r["new_unseen_dev_historical_label_SHA256"],
          "metrics":r["metrics"]},sort_keys=True),flush=True)

if __name__=="__main__":main()
