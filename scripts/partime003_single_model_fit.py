"""PAR-TIME-003 genuine-source single-arm five-seed developmental FIT.

This is deliberately a SINGLE NEURAL ARCHITECTURE study. PAR-TIME-002's
complete cross-architecture training-FLOP parity is OPEN, so NO comparative
claim against time-blind TAC, GRU, SSM, Transformer or equal budget is made.

All physical observations from original SHA-verified UCI PRSA2017 2013-2014
source; only March-April 2014 source histories/targets train models; May-June
2014 is historically EXPOSED development, not independent qualification.
No 2015/2016/2017 numeric rows are parsed and no target crosses the model
forward path. Original model, optimizer, RNG, sample order and site scores
are frozen for clean-checkout replay. No action/robot control is trained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from scripts.partime002_real_source import (
    SHA256, read_source, chronology, batch_from_cases, score
)
from tac_osm.variants.parallel_time_age_world import TimeAwareParallelResidualWorld

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/"research/parallel-six/PAR-TIME-003-PROTOCOL.json"
PLAN=ROOT/"research/parallel-six/PAR-TIME-003-FROZEN-PLAN.json"
OUT=ROOT/"research/replay-bundles/PAR-TIME-003"
STEPS=64
BATCH=4
SEEDS=(271028,271029,271030,271031,271032)
LR=0.0003

def file_hash(path:Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,
                          separators=(",",":")).encode()).hexdigest()

def modelspec(seed):
    torch.manual_seed(seed)
    model=TimeAwareParallelResidualWorld()
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params!=96639 or any(p.dtype!=torch.float32 for p in model.parameters()):
        raise RuntimeError("source neural model active geometry changed")
    return model

def schedule(train,seed):
    rng=random.Random(seed)
    return [list(train[rng.randrange(len(train))]) for _ in range(STEPS*BATCH)]

def source_cases(src):
    cfg=json.loads(PROTOCOL.read_text())
    if (cfg["study_id"]!="PAR-TIME-003-SINGLE-ARM-REAL-DEV"
        or cfg["status"]!="SINGLE_ARM_DEVELOPMENT_NOT_ARCHITECTURE_COMPARISON"
        or cfg["source_original_zip_SHA256"]!=SHA256
        or cfg["optimizer_steps_per_model"]!=STEPS
        or cfg["batch"]!=BATCH or cfg["lr"]!=LR
        or cfg["seeds"]!=list(SEEDS)
        or cfg["active_model_parameters"]!=96639):
        raise RuntimeError("original source experiment registration changed")
    d=read_source(src)
    train,eval_rows=chronology(d)
    if max(t for _,_,t in train)+1>=min(t for _,_,t in eval_rows):
        raise RuntimeError("true source timeline contamination")
    return d,train,eval_rows

def plan(src):
    if PLAN.exists() or OUT.exists():
        raise RuntimeError("original source plan/model already exists; no overwrites")
    _,train,dev=source_cases(src)
    p={
       "study_id":"PAR-TIME-003-SINGLE-ARM-REAL-DEV",
       "source_ZIP_SHA256":SHA256,
       "original_training_case_count":len(train),
       "original_development_case_count":len(dev),
       "original_training_case_SHA256":digest(train),
       "previously_exposed_development_case_SHA256":digest(dev),
       "development_timestamp_cases_only":dev,
       "sampling_orders":{str(seed):schedule(train,seed) for seed in SEEDS},
       "per_seed_sample_order_SHA256":
         {str(seed):digest(schedule(train,seed)) for seed in SEEDS},
       "optimizer_steps_per_model":STEPS,"batch":BATCH,
       "no_model_training_occurred_before_PLAN_commit":True,
       "no_independent_holdout_or_architecture_comparison":True
    }
    PLAN.write_text(json.dumps(p,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("PAR_TIME_003_PLAN_TO_BE_COMMITTED_BEFORE_FIT="+json.dumps({
        "source_SHA":SHA256,
        "train_cases":len(train),"already_exposed_dev_cases":len(dev),
        "seeds":list(SEEDS),"sample_order_hashes":p["per_seed_sample_order_SHA256"],
        "new_models_trained":0},sort_keys=True),flush=True)

def freeze(src):
    if not PLAN.exists():
        raise RuntimeError("source sample order must already be committed")
    _,train,dev=source_cases(src)
    p=json.loads(PLAN.read_text())
    if (p["source_ZIP_SHA256"]!=SHA256
        or p["original_training_case_SHA256"]!=digest(train)
        or p["previously_exposed_development_case_SHA256"]!=digest(dev)
        or p["development_timestamp_cases_only"]!=[list(x) for x in dev]
        or p["optimizer_steps_per_model"]!=STEPS
        or p["batch"]!=BATCH):
        raise RuntimeError("source/split changed from precommitted plan")
    for seed in SEEDS:
        if (p["sampling_orders"][str(seed)]!=schedule(train,seed)
            or p["per_seed_sample_order_SHA256"][str(seed)]!=digest(schedule(train,seed))):
            raise RuntimeError("committed source sample schedule changed")
    return p

def predict(model,inputs):
    obs,covariates,elapsed,writes,target,valid=inputs
    y=model(obs,covariates,elapsed_dt=elapsed,write_mask=writes)[:,-1]
    return y

def fit(src):
    if OUT.exists():
        raise RuntimeError("immutable original training checkpoint directory exists")
    d,_,_=source_cases(src)
    p=freeze(src)
    OUT.mkdir(parents=True,exist_ok=False)
    traces=[]
    for seed in SEEDS:
        model=modelspec(seed)
        optimizer=torch.optim.AdamW(model.parameters(),lr=LR,
                      weight_decay=0.001,foreach=False,fused=False)
        first=[v.detach().clone() for v in model.parameters()]
        rngplan=p["sampling_orders"][str(seed)]
        losses=[]
        began=time.monotonic()
        model.train()
        for step_i in range(STEPS):
            rows=rngplan[step_i*BATCH:(step_i+1)*BATCH]
            inputs=batch_from_cases(d,rows)
            optimizer.zero_grad(set_to_none=True)
            actual=predict(model,inputs)
            loss=score(actual,inputs[-2],inputs[-1])
            if not torch.isfinite(loss).item():
                raise RuntimeError("nonfinite actual physical training loss")
            loss.backward()
            if any(v.grad is None or not v.grad.isfinite().all().item()
                   for v in model.parameters()):
                raise RuntimeError("disconnected original training parameter")
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        if not any(not torch.equal(x,y)
                   for x,y in zip(first,model.parameters())):
            raise RuntimeError("full training did not change any neural parameter")
        checkpoint=OUT/f"seed-{seed}-full-original.pt"
        torch.save({
          "study_id":"PAR-TIME-003-SINGLE-ARM-REAL-DEV",
          "architecture":"time_aware_parallel_Tac_postscan_FFN",
          "source_sha256":SHA256,
          "seed":seed,"optimizer_steps":STEPS,"batch":BATCH,
          "model_state":model.state_dict(),
          "optimizer_state":optimizer.state_dict(),
          "torch_rng":torch.get_rng_state(),
          "python_rng":random.getstate(),
          "fixed_train_case_SHA256":p["original_training_case_SHA256"],
          "fixed_dev_case_SHA256":p["previously_exposed_development_case_SHA256"],
          "full_sample_order_SHA256":p["per_seed_sample_order_SHA256"][str(seed)],
          "original_training_loss_trace":losses,
          "training_model_source_SHA256":file_hash(
             ROOT/"src/tac_osm/variants/parallel_time_age_world.py"),
          "source_loader_SHA256":file_hash(
             ROOT/"scripts/partime002_real_source.py"),
          "trainer_SHA256":file_hash(
             ROOT/"scripts/partime003_single_model_fit.py"),
          "trained_solely_on_original_real_source":True,
          "independently_qualified":False
        },checkpoint)
        traces.append({
            "seed":seed,"checkpoint_SHA256":file_hash(checkpoint),
            "training_first_step_loss":losses[0],
            "training_final_step_loss":losses[-1],
            "training_last_16_mean":sum(losses[-16:])/16,
            "optimizer_steps":STEPS,"physical_source_examples_seen":STEPS*BATCH,
            "trained_parameters":96639,
            "elapsed_cpu_seconds_outcome_only":time.monotonic()-began
        })
        print("PAR_TIME_003_FINISHED_GENUINE_SOURCE_SEED="+json.dumps(
            traces[-1],sort_keys=True),flush=True)
    receipt={
        "study_id":"PAR-TIME-003-SINGLE-ARM-REAL-DEV",
        "status":"ORIGINAL_REAL_TRAINED_CHECKPOINTS_BEFORE_DEV",
        "actual_model_count":len(traces),
        "original_source_sha256":SHA256,
        "source_train_cases_count":p["original_training_case_count"],
        "source_dev_cases_count":p["original_development_case_count"],
        "development_original_numerics_scored":False,
        "full_training_FLops_matched_architecture_comparison":False,
        "no_comparator_model_was_trained":True,
        "no_architecture_superiority_claim":True,
        "original_checkpoints":traces
    }
    (OUT/"TRAIN_RECEIPT.json").write_text(
       json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("PAR_TIME_003_ALL_5_ORIGINAL_SOURCE_MODELS_FROZEN_BEFORE_DEV",flush=True)

def all_predictions(model,d,rows):
    batches=[]
    with torch.no_grad():
        model.eval()
        for start in range(0,len(rows),BATCH):
            subset=rows[start:start+BATCH]
            inp=batch_from_cases(d,subset)
            pred=predict(model,inp)
            baseline=inp[0][:,-1]
            batches.append((pred.detach(),inp[-2],inp[-1],baseline))
    return tuple(torch.cat([b[i] for b in batches],0) for i in range(4))

def result_metrics(values,center,spread):
    pred,real,valid,base=values
    ok=valid[:,:13].float()
    n=int(ok.sum().item())
    if n<50:
        raise RuntimeError("insufficient original physical outcomes")
    abs_score=((pred[:,:13]-real[:,:13]).abs()*ok).sum()/n
    abs_base=((base[:,:13]-real[:,:13]).abs()*ok).sum()/n
    pm=valid[:,0]
    if not pm.any().item(): raise RuntimeError("no real PM2.5 next-hour targets")
    def physical(y):
        return torch.expm1((y*spread[0]+center[0]).clamp(0.,12.)).clamp_min(0.)
    pmae=float((physical(pred[:,0])[pm]-physical(real[:,0])[pm]).abs().mean())
    bmae=float((physical(base[:,0])[pm]-physical(real[:,0])[pm]).abs().mean())
    return dict(
      authentic_examples=pred.shape[0],observed_physical_channels=n,
      observed_pm25_future_targets=int(pm.sum().item()),
      normalized_13_channel_MAE=float(abs_score),
      real_observed_persistence_13_channel_MAE=float(abs_base),
      PM25_ug_m3_MAE=pmae,
      PM25_persistence_ug_m3_MAE=bmae,
      genuine_MAE_difference_from_persistence=float(abs_score-abs_base),
      nominal_90_coverage="NOT_MODELED",
      independent_sealed_test="NOT_EVALUATED",
      action_causal_skill="NOT_TESTED"
    )

def check_original_weights(receipt,p):
    rows=receipt["original_checkpoints"]
    if len(rows)!=5 or set(q["seed"] for q in rows)!=set(SEEDS):
        raise RuntimeError("original 5 model checkout incomplete")
    for record in rows:
        checkpoint=OUT/f"seed-{record['seed']}-full-original.pt"
        if file_hash(checkpoint)!=record["checkpoint_SHA256"]:
            raise RuntimeError("frozen genuine trained checkpoint hash mismatch")
        st=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if (st["source_sha256"]!=SHA256
           or st["full_sample_order_SHA256"]!=p[
                "per_seed_sample_order_SHA256"][str(record["seed"])]
           or st["optimizer_steps"]!=STEPS
           or st["trained_solely_on_original_real_source"] is not True
           or st["training_model_source_SHA256"]!=file_hash(
              ROOT/"src/tac_osm/variants/parallel_time_age_world.py")
           or st["source_loader_SHA256"]!=file_hash(
              ROOT/"scripts/partime002_real_source.py")
           or st["trainer_SHA256"]!=file_hash(
              ROOT/"scripts/partime003_single_model_fit.py")):
            raise RuntimeError("original model/source/optimizer freeze invalid")

def evaluate(src,*,replay=False):
    d,_,dev=source_cases(src)
    p=freeze(src)
    receipt=json.loads((OUT/"TRAIN_RECEIPT.json").read_text())
    check_original_weights(receipt,p)
    original_scores_path=OUT/"DEV_RESULTS.json"
    if not replay and original_scores_path.exists():
        raise RuntimeError("source dev score already exists: never rescore")
    if replay and not original_scores_path.exists():
        raise RuntimeError("original source development score receipt missing")
    rows=[]
    for record in receipt["original_checkpoints"]:
        seed=record["seed"]
        st=torch.load(OUT/f"seed-{seed}-full-original.pt",
                      map_location="cpu",weights_only=False)
        model=modelspec(seed)
        model.load_state_dict(st["model_state"],strict=True)
        original=all_predictions(model,d,dev)
        metrics=result_metrics(original,d["center"],d["spread"])
        if replay:
            old=torch.load(OUT/f"seed-{seed}-DEV_arrays.pt",
                           map_location="cpu",weights_only=True)
            for idx,(now,prev) in enumerate(zip(original,old)):
                torch.testing.assert_close(now,prev,rtol=1e-5,atol=1e-6,
                           msg=f"first source DEV forecast differs array {idx} seed {seed}")
        else:
            torch.save(original,OUT/f"seed-{seed}-DEV_arrays.pt")
        rows.append(dict(seed=seed,original_checkpoint_sha=record["checkpoint_SHA256"],
                         **metrics))
    report=dict(
       study_id="PAR-TIME-003-SINGLE-ARM-REAL-DEV",
       source_original_SHA256=SHA256,
       actual_train_models=5,
       original_dev_cohort_SHA256=p["previously_exposed_development_case_SHA256"],
       already_exposed_2014_development=True,
       scientific_qualification="NOT_QUALIFIED",
       matched_learned_architecture_comparison=False,
       trained_real_source_not_independent_holdout=True,
       no_protected_2015_2017_numeric_rows_read=True,
       record_each_seed=rows)
    if replay:
        original_report=json.loads(original_scores_path.read_text())
        for a,b in zip(original_report["record_each_seed"],rows):
            for k in a:
                if isinstance(a[k],(float,int)) and not isinstance(a[k],bool):
                    if abs(a[k]-b[k])>1e-6:raise RuntimeError("original DEV metric replay changed "+k)
                elif a[k]!=b[k]:
                    raise RuntimeError("original DEV report identity drift "+k)
        print("PAR_TIME_003_INDEPENDENT_ORIGINAL_SOURCE_DEV_REPLAY_PASS",flush=True)
    else:
        original_scores_path.write_text(
            json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
        print("PAR_TIME_003_ORIGINAL_SOURCE_DEV_RESULTS="+json.dumps(
            report,sort_keys=True,allow_nan=False),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--mode",choices=("plan","fit","eval","replay"),required=True)
    args=ap.parse_args()
    torch.set_num_threads(1)
    if args.mode=="plan":plan(args.source)
    elif args.mode=="fit":fit(args.source)
    elif args.mode=="eval":evaluate(args.source)
    else:evaluate(args.source,replay=True)

if __name__=="__main__":
    main()
