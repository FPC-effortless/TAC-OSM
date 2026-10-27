"""LATFUSE018 exploratory REAL physical-source training, never protected test.

Exactly matched source time, data, masks, task, seed, batch, history and
AdamW for original 99,390-weight/6,800B associative vs fastslow historic
queries. 64 updates each arm/seed. Precommitted PLAN.json BEFORE first fit.
Original UCI501 May-Jun FIT rows repeatedly analyzed in other experiments,
so this is development evidence, NOT independent qualification.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import random
import time
from pathlib import Path
import torch
from torch.nn import functional as F

from scripts import latfuse002_real_fit as source
from scripts import latfuse008_paired_real_training as original
from scripts.latfuse016_authentic_content_nofit import original_source_forward_loss
from scripts.validate_research_parity import check_manifest

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse018/REGISTERED_REAL_FIT.json"
PLAN=ROOT/"research/mathematical-engineering/latfuse018/PLAN.json"
PARITY=ROOT/"research/mathematical-engineering/latfuse017/MATH_PARITY.json"
OUT=ROOT/"research/replay-bundles/LATFUSE018"
SEEDS=(18018,18019)
ARMS=("assoc_historic_query","fast_slow_historic_query")
STEPS=64
BATCH=16
TRAIN_START=source.hour(dt.datetime(2014,3,1))
TRAIN_STOP=source.hour(dt.datetime(2014,4,18))
EVAL_START=source.hour(dt.datetime(2014,5,1))
EVAL_STOP=source.hour(dt.datetime(2014,6,10))
LAGS=(16,64,128,512)
HISTORY=24
MAXLAG=512
LR=.0005

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest_rows(z):return hashlib.sha256(json.dumps(z,separators=(",",":")).encode()).hexdigest()
def source_rows(d,start,stop,every):
    out=[]
    for s in range(12):
        for t in range(start,stop,every):
            if (t-HISTORY+1>=0 and t+MAXLAG+6<source.hour(source.TRAIN_END)
                and bool(d["valid"][s,t,0])
                and bool(d["valid"][s,t+64+6].any())):
                out.append((s,t))
    if len(out)<100 or len(set(x[0] for x in out))<10:
        raise RuntimeError("insufficient station-covered real/source valid PM2.5 cases")
    return out

def verify_source(arg):
    if sha(arg)!=source.SOURCE_SHA:raise RuntimeError("original real UCI501 zip source SHA changed")
    r=json.loads(REG.read_text())
    if (r["study_type"]!="single_model_training" or not r["registered_before_first_fit"]
        or r["study_id"]!="LATFUSE018-CONTENT-RECONSTRUCTION-EXPLORATORY-REAL"
        or r["training_optimizer"]["steps"]!=STEPS
        or r["training_optimizer"]["batch"]!=BATCH
        or r["training_optimizer"]["lr"]!=LR):
        raise RuntimeError("physical training registration differs from model study")
    check_manifest(r,"preflight")
    proof=json.loads(PARITY.read_text())
    if (proof["registered_shape_full_operator_mathematical_FLOP_parity_within_1pct"] is not True
        or not all(x["strictly_conservative_relative_math_FLOP_gap_upper_bound"]<.01
            for x in proof["each_seed"])):
        raise RuntimeError("original complete ATen math FLOP parity prerequisite missing")
    d=original.source_observations(arg)
    train=source_rows(d,TRAIN_START,TRAIN_STOP,12)
    evalrows=source_rows(d,EVAL_START,EVAL_STOP,96)
    if max(t for _,t in train)+70>=min(t for _,t in evalrows):
        raise RuntimeError("training future +6 could overlap source developmental evaluation period")
    return r,d,train,evalrows,proof

def schedule(train,seed):
    g=random.Random(seed)
    return [list(train[g.randrange(len(train))]) for _ in range(STEPS*BATCH)]

def produce_plan(source_path):
    _,_,train,evalrows,proof=verify_source(source_path)
    if PLAN.exists() or OUT.exists():raise RuntimeError("fit plan or model already present")
    ordered={str(s):schedule(train,s) for s in SEEDS}
    plan=dict(study_id="LATFUSE018-CONTENT-RECONSTRUCTION-EXPLORATORY-REAL",
        source_zip_SHA256=source.SOURCE_SHA,
        true_source_training_rows=len(train),true_source_eval_rows=len(evalrows),
        true_source_training_case_SHA256=digest_rows(train),
        exposed_source_development_eval_case_SHA256=digest_rows(evalrows),
        training_rows=train,development_rows=evalrows,
        full_training_orders=ordered,
        each_seed_sample_order_sha256={s:digest_rows(v) for s,v in ordered.items()},
        original_2_seed_2_arm_matrix=True,
        per_arm_steps=STEPS,per_arm_batch=BATCH,recorded_history_hours=88,
        original_params=99390,original_recurrent_bytes=6800,
        original_math_FLOP_registered_upper_diff_bound=max(
            q["strictly_conservative_relative_math_FLOP_gap_upper_bound"]
            for q in proof["each_seed"]),
        original_past_PM25_only_source_after_query=True,
        evaluation_source_exposed_and_NOT_virgin=True,
        no_protected_2015_2017_numerics=true)
    PLAN.write_text(json.dumps(plan,indent=2,sort_keys=True)+"\n")
    print("LATFUSE018_REGISTERED_BEFORE_FIRST_NEURAL_FIT="+json.dumps({
        "source_train_rows":len(train),"source_eval_rows":len(evalrows),
        "train_case_sha":plan["true_source_training_case_SHA256"],
        "eval_case_sha":plan["exposed_source_development_eval_case_SHA256"],
        "training_order_shas":plan["each_seed_sample_order_sha256"],
        "parity_upper_bound":plan["original_math_FLOP_registered_upper_diff_bound"]
      },sort_keys=True),flush=True)

def frozen_plan(train,evalrows):
    if not PLAN.exists():raise RuntimeError("precommitted genuine source train order not found")
    p=json.loads(PLAN.read_text())
    if (p["source_zip_SHA256"]!=source.SOURCE_SHA
        or p["per_arm_steps"]!=STEPS or p["per_arm_batch"]!=BATCH
        or p["true_source_training_case_SHA256"]!=digest_rows(train)
        or p["exposed_source_development_eval_case_SHA256"]!=digest_rows(evalrows)
        or p["training_rows"]!=[list(x) for x in train]
        or p["development_rows"]!=[list(x) for x in evalrows]):
        raise RuntimeError("genuine original historical source split/registered plan changed")
    for seed in SEEDS:
        if (p["full_training_orders"][str(seed)]!=schedule(train,seed)
          or p["each_seed_sample_order_sha256"][str(seed)]!=digest_rows(schedule(train,seed))):
            raise RuntimeError("registered sample source order changed")
    return p

def scores(model,d,cases,lag):
    # Genuine recorded physical observations: forecast labels never query inputs.
    scores=dict(n=0,eligible=0,content_norm_L1_sum=0.,content_log_L1_sum=0.,
       content_raw_L1_sum=0.,source_prior_norm_L1_sum=0.,
       current_PM_persistence_log_L1_sum=0.,
       zero_read_historic_norm_L1_sum=0.,
       future_eligible=0,forecast_abs_sum=0.,persist_abs_sum=0.,
       future_nll_sum=0.,future_90_coverage_sum=0.)
    model.eval()
    with torch.no_grad():
        for i in range(0,len(cases),BATCH):
            segment=cases[i:i+BATCH]
            site=torch.tensor([row[0] for row in segment])
            anchor=torch.tensor([row[1] for row in segment])
            s=model.initial(len(segment))
            for offset in range(-HISTORY+1,lag+1):
                t=anchor+offset
                s=model.step(s,d["values"][site,t],d["valid"][site,t],
                     d["age"][site,t],d["last"][site,t],site)
            clock=source.calendar([int(x) for x in anchor])[:,None,:]
            predicted=model.historic_event_logits(s,site,clock)[:,0]
            # Counterfactual: remove all original associative matrix
            # for historic scalar read only, after full legally observed stream.
            zero=s._replace(associative=torch.zeros_like(s.associative))
            ablated=model.historic_event_logits(zero,site,clock)[:,0]
            future=source.calendar([int(x+lag+6) for x in anchor])[:,None,:]
            mean,sd=model.forecast(s,site,future)
            # Only after model outputs access original physically measured targets.
            y=d["values"][site,anchor,0]
            avail=d["valid"][site,anchor,0]
            if not bool(avail.all()):raise RuntimeError("preregistered original source target missing")
            original_log_true=y*d["spread"][0]+d["center"][0]
            original_log_pred=predicted*d["spread"][0]+d["center"][0]
            log_last=d["last"][site,anchor+lag+1,0]*d["spread"][0]+d["center"][0]
            scores["n"]+=len(segment);scores["eligible"]+=int(avail.sum())
            scores["content_norm_L1_sum"]+=float((predicted-y).abs().sum())
            scores["content_log_L1_sum"]+=float((original_log_pred-original_log_true).abs().sum())
            scores["content_raw_L1_sum"]+=float((torch.expm1(original_log_pred.clamp(0,10))-
                    torch.expm1(original_log_true.clamp(0,10))).abs().sum())
            scores["source_prior_norm_L1_sum"]+=float(y.abs().sum())
            scores["current_PM_persistence_log_L1_sum"]+=float((log_last-original_log_true).abs().sum())
            scores["zero_read_historic_norm_L1_sum"]+=float((ablated-y).abs().sum())
            mu=mean[:,0,:];sigma=sd[:,0,:]
            yt=d["values"][site,anchor+lag+6]
            ok=d["valid"][site,anchor+lag+6]
            yt=torch.where(ok,yt,mu)
            present=ok.to(mu.dtype)
            scores["future_eligible"]+=int(ok.sum())
            scores["forecast_abs_sum"]+=float(((mu-yt).abs()*present).sum())
            latest=d["last"][site,anchor+lag+1]
            scores["persist_abs_sum"]+=float(((latest-yt).abs()*present).sum())
            nll=.5*(math.log(2*math.pi)+2*torch.log(sigma)+((yt-mu)/sigma).square())
            scores["future_nll_sum"]+=float((nll*present).sum())
            coverage=((yt-mu).abs()<=1.6448536269514722*sigma).float()
            scores["future_90_coverage_sum"]+=float((coverage*present).sum())
    n=scores["eligible"];f=scores["future_eligible"]
    if n<20 or f<100:raise RuntimeError("invalid real source development evaluation denominator")
    return dict(original_real_source_historical_target_examples=n,
        physically_observed_PM25_log1p_MAE=scores["content_log_L1_sum"]/n,
        physically_observed_PM25_raw_ugm3_MAE=scores["content_raw_L1_sum"]/n,
        original_normalized_historical_PM25_MAE=scores["content_norm_L1_sum"]/n,
        original_past_source_train_mean_normalized_MAE=scores["source_prior_norm_L1_sum"]/n,
        actual_current_last_PM25_log1p_persistence_MAE=scores["current_PM_persistence_log_L1_sum"]/n,
        zero_original_matrix_historic_normalized_MAE=scores["zero_read_historic_norm_L1_sum"]/n,
        real_source_physically_observed_future_channels=f,
        original_source_future_normalized_MAE=scores["forecast_abs_sum"]/f,
        original_source_future_persistence_MAE=scores["persist_abs_sum"]/f,
        original_source_future_gaussian_NLL=scores["future_nll_sum"]/f,
        original_source_future_nominal_90_coverage=scores["future_90_coverage_sum"]/f)

def fitting(src):
    if OUT.exists():raise RuntimeError("original trained result exists; DO NOT OVERWRITE")
    _,d,train,evalrows,proof=verify_source(src)
    p=frozen_plan(train,evalrows)
    OUT.mkdir(parents=True,exist_ok=False)
    outcomes=[]
    for seed in SEEDS:
        samples=p["full_training_orders"][str(seed)]
        for name in ARMS:
            model=original.create_model(seed,name)
            opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.001,
                                  foreach=False,fused=False)
            initial=[q.detach().clone() for q in model.parameters()]
            losses=[]
            began=time.monotonic()
            for i in range(STEPS):
                case=samples[i*BATCH:(i+1)*BATCH]
                opt.zero_grad(set_to_none=True)
                model.train()
                value,cont,fore,h=original_source_forward_loss(model,d,case)
                if not bool(torch.isfinite(value)):
                    raise RuntimeError("nonfinite genuine source B0 training loss")
                value.backward()
                if any(param.grad is None or not torch.isfinite(param.grad).all()
                       for param in model.parameters()):
                    raise RuntimeError("invalid original trained source gradients")
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
                losses.append([float(value.detach()),float(cont.detach()),float(fore.detach())])
                if i%16==0:
                    print("LATFUSE018_REAL_SOURCE_TRAINING="+json.dumps({
                        "seed":seed,"arm":name,"update":i,"objective":losses[-1],
                        "exploratory_source_only":True},sort_keys=True),flush=True)
            if not any(not torch.equal(x,y) for x,y in zip(initial,model.parameters())):
                raise RuntimeError("original source neural model did NOT learn")
            model.eval()
            dev={str(lag):scores(model,d,evalrows,lag) for lag in LAGS}
            folder=OUT/f"seed-{seed}"/name
            folder.mkdir(parents=True)
            checkpoint=folder/"final.pt"
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),
               torch_rng=torch.get_rng_state(),source_real_zip_SHA256=source.SOURCE_SHA,
               seed=seed,architecture=name,steps=STEPS,examples_seen=STEPS*BATCH,
               sample_order_SHA256=p["each_seed_sample_order_sha256"][str(seed)],
               dev_original_source_cohort_SHA256=p["exposed_source_development_eval_case_SHA256"],
               training_loss_log=losses,eval_development_scores=dev,
               model_source_script_SHA256=sha(ROOT/"scripts/latfuse018_real_content_training.py"),
               original_99390_parameter_count=99390,
               original_per_stream_6800_nominal_FP32_bytes=True,
               trained_on_authentic_source=True,
               genuine_development_eval_PREVIOUSLY_EXPOSED=True),checkpoint)
            obs=dict(seed=seed,architecture=name,original_checkpoint_SHA256=sha(checkpoint),
                original_source_training_sample_order_SHA256=p["each_seed_sample_order_sha256"][str(seed)],
                optimizer_steps=STEPS,examples_seen=STEPS*BATCH,
                initial_training_total_loss=losses[0][0],
                final_training_total_loss=losses[-1][0],
                final_16_training_total_losses_mean=sum(x[0] for x in losses[-16:])/16,
                CPU_elapsed_seconds=time.monotonic()-began,
                original_live_params=99390,nominal_recurrent_bytes=6800,
                original_actual_source_development_eval=dev)
            outcomes.append(obs)
            print("LATFUSE018_REAL_PHYSICAL_TRAINED="+json.dumps(obs,sort_keys=True),flush=True)
    out=dict(study_id=p["study_id"],
       source_SHA256=source.SOURCE_SHA,
       original_train_case_SHA256=p["true_source_training_case_SHA256"],
       exposed_development_source_case_SHA256=p["exposed_source_development_eval_case_SHA256"],
       real_training_examples=len(train),exposed_development_source_cases=len(evalrows),
       original_2014FIT_uses_authentic_physical_source=True,
       actual_model_seed_count=2,actual_architecture_count=2,optimizer_updates_total=STEPS*4,
       original_historical_architecture_comparison_graph_H64_batch16_math_FLOP_discrepancy_upper=p["original_math_FLOP_registered_upper_diff_bound"],
       claim_full_hardware_FLOP_or_new_shape_parity=False,
       independent_validated_model_superiority_QUALIFIED=False,
       original_development_MayJun_already_exposed=True,
       original_numerical2015_exposed_and_protected2016_17_unopened=True,
       source_owner_and_target_only_after_prediction=True,
       no_synthetic_performance_claim=True,
       all_checkpoints_include_optimizer_RNG_and_exact_source_order=True,
       models=outcomes)
    (OUT/"RESULTS.json").write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("LATFUSE018_REAL_TRAINING_DONE="+json.dumps({
       "models":len(outcomes),"updates_each":STEPS,"exploratory_not_qualified":True
    },sort_keys=True),flush=True)

def replay(src):
    _,d,train,evalrows,_=verify_source(src)
    p=frozen_plan(train,evalrows)
    out=json.loads((OUT/"RESULTS.json").read_text())
    if len(out["models"])!=4 or out["optimizer_updates_total"]!=STEPS*4:
        raise RuntimeError("original FIT model training receipt incomplete")
    for row in out["models"]:
        path=OUT/f"seed-{row['seed']}"/row["architecture"]/"final.pt"
        if sha(path)!=row["original_checkpoint_SHA256"]:
            raise RuntimeError("original source real model checkpoint hash mismatch")
        st=torch.load(path,map_location="cpu",weights_only=False)
        if (st["sample_order_SHA256"]!=p["each_seed_sample_order_sha256"][str(row["seed"])]
            or st["steps"]!=STEPS or st["seed"]!=row["seed"] or
            st["source_real_zip_SHA256"]!=source.SOURCE_SHA or
            st["model_source_script_SHA256"]!=sha(ROOT/"scripts/latfuse018_real_content_training.py")):
            raise RuntimeError("source data/optimizer or code replay integrity mismatch")
        model=original.create_model(row["seed"],row["architecture"])
        model.load_state_dict(st["model"],strict=True)
        model.eval()
        for lag in LAGS:
            again=scores(model,d,evalrows,lag)
            past=st["eval_development_scores"][str(lag)]
            for k in again:
                # Exact Torch CPU deterministic inference and dataloader.
                if isinstance(again[k],float):
                    if abs(again[k]-past[k])>1e-7:
                        raise RuntimeError("source trained checkpoint independent deterministic replay mismatch")
                elif again[k]!=past[k]:
                    raise RuntimeError("source model true label count/source replay mismatch")
        print("LATFUSE018_REAL_SOURCE_CHECKPOINT_REPLAY_PASS="+json.dumps({
           "seed":row["seed"],"arm":row["architecture"],
           "original_checkpoint_sha256":sha(path)},sort_keys=True))
    print("LATFUSE018_REAL_ALL_SOURCE_CHECKPOINT_REPLAY_SUCCESS")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("plan","fit","replay"),required=True)
    p.add_argument("--source",type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    if a.mode=="plan":produce_plan(a.source)
    elif a.mode=="fit":fitting(a.source)
    else:replay(a.source)
if __name__=="__main__":main()
