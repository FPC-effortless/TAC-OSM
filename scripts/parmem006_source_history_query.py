"""PAR-MEM-006 authentic historic PM2.5 source retrieval from TAC state.

A query asks for a genuinely PREVIOUSLY OBSERVED station event's PM2.5
measurement at a source-known timestamp, lag 16/64/128 recorded events.
The past value was observed by TAC during history ingestion, but is never
passed through the retrieval query/head input. The source target is read
only after the state and query features have been generated.

Both retrieval heads have EXACT same active parameter count and graph
shape. Associative intervention gates frozen matrix read at the same
place; query key also enters the head, so no dummy trainable weights.
The source-grounded pretrained 96,639 TAC core stays FROZEN, originally
trained only on genuine air data in PAR-TIME-005.

This is an EXPLORATORY real-source probe, not a qualified architecture
win: full forward/back/optimizer FLOP parity remains UNVERIFIED, and
May-Jun 2014 genuine development data is previously exposed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from scripts.partime002_real_source import SHA256, read_source
from scripts.partime005_continuous_real_training import (
    source_and_times, frozen, verify_originals, observed_source_step,
    model_for, SEEDS
)

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/parallel-six/PAR-MEM-006-PROTOCOL.json"
PLAN=ROOT/"research/parallel-six/PAR-MEM-006-FROZEN-PLAN.json"
OUT=ROOT/"research/replay-bundles/PAR-MEM-006"
SOURCE_ROOT=ROOT/"research/replay-bundles/PAR-TIME-005"
ARMS=("associative_read","history_blind")
LAGS=(16,64,128)
START=144
STRIDE=8
TRAIN_STEPS=64
BATCH=64
LR=.002

def checksum(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()

class SourceHistoricQueryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_key=nn.Linear(8,8)
        self.head=nn.Sequential(nn.Linear(88+14+8+8,32),nn.SiLU(),nn.Linear(32,1))

    def forward(self,summary,matrix,present,metadata,*,associative):
        if (summary.shape[-1]!=88 or matrix.shape[-2:]!=(8,8)
             or present.shape[-1]!=14 or metadata.shape[-1]!=8):
            raise ValueError("typed source query geometry changed")
        q=F.normalize(self.query_key(metadata),dim=-1,eps=1e-6)
        # Both arms perform same trained operator shapes. The zero-memory
        # control's q is STILL fed directly to the head and receives
        # gradients, so there are no unused/dummy learned parameters.
        read=((matrix*float(bool(associative)))@q.unsqueeze(-1)).squeeze(-1)
        features=torch.cat((summary,present,read,q),dim=-1)
        return self.head(features).squeeze(-1)

def model_pair(seed,mode):
    torch.manual_seed(seed+65000)
    head=SourceHistoricQueryHead()
    if mode not in ARMS:raise RuntimeError("unregistered real-source query arm")
    n=sum(q.numel() for q in head.parameters() if q.requires_grad)
    if n!=3913 or n+96639!=100552:
        raise RuntimeError("live 100k deployed weight budget failed")
    if any(p.dtype!=torch.float32 for p in head.parameters()):
        raise RuntimeError("source model cannot run mixed precision unknown")
    return head

def original_frozen_core(seed,receipt):
    row=next((x for x in receipt["original_checkpoints"] if x["seed"]==seed),None)
    if row is None:raise RuntimeError("trained original seed missing")
    path=SOURCE_ROOT/f"seed-{seed}-original-continuous.pt"
    if checksum(path)!=row["original_checkpoint_SHA256"]:
        raise RuntimeError("original 768 event recurrent state checkpoint SHA drift")
    state=torch.load(path,map_location="cpu",weights_only=False)
    if (state["publisher_source_sha256"]!=SHA256
        or state["continuous_genuine_observed_events_per_station"]!=768
        or not state["retained_state_not_reset_between_fit_updates"]):
        raise RuntimeError("pretrained genuine continuous source origin wrong")
    core=model_for(seed)
    core.load_state_dict(state["model_state"],strict=True)
    core.eval()
    for p in core.parameters():p.requires_grad_(False)
    return core

def source_query_metadata(clocks,target_clock,target_index,current_index,lag):
    # Only source owner’s timestamps/sequence ordinals are provided:
    # NOT the recorded measurement at target, future targets or oracle key.
    time=torch.tensor(target_clock,dtype=torch.float32)
    idx=torch.tensor(target_index,dtype=torch.float32)
    current=torch.tensor(current_index,dtype=torch.float32)
    distance=torch.full_like(time,float(lag))
    return torch.stack((
       torch.sin(2*math.pi*time/24),torch.cos(2*math.pi*time/24),
       torch.sin(2*math.pi*time/168),torch.cos(2*math.pi*time/168),
       torch.sin(idx/16),torch.cos(idx/16),
       torch.log1p(distance),torch.log1p(current)
    ),dim=-1)

def timestamps(fit,dev):
    def construct(rows):
        # site-dependent source clocks are in increasing observed source
        # order, anchors chosen by original source order alone.
        records=[]
        shortest=min(len(x) for x in rows)
        for anchor in range(START,shortest,STRIDE):
            for lag in LAGS:
                for site in range(12):
                    older=rows[site][anchor-lag]
                    now=rows[site][anchor]
                    if older>=now:
                        raise RuntimeError("source query is not for a PAST observed event")
                    records.append((site,anchor,lag,older,now))
        return records
    return construct(fit),construct(dev)

def features_from_timed_stream(seed,rows,queries,receipt,data):
    core=original_frozen_core(seed,receipt)
    groups={}
    for j,(site,anchor,lag,older,now) in enumerate(queries):
        groups.setdefault(anchor,[]).append((j,site,lag,older,now))
    inputs=[None]*len(queries)
    state=core.init_stream(12)
    before=None
    with torch.no_grad():
        for anchor in range(min(len(x) for x in rows)):
            clocks=[r[anchor] for r in rows]
            source_obs,available,dt,admit=observed_source_step(data,clocks,before)
            _,state=core.stream_step(state,source_obs,available,
                                     elapsed_dt=dt,write_mask=admit)
            before=clocks
            if anchor not in groups: continue
            base=state.fast_sum/state.count
            for j,site,lag,older,now in groups[anchor]:
                if rows[site][anchor-lag]!=older or rows[site][anchor]!=now:
                    raise RuntimeError("source label clock mismatch")
                query=source_query_metadata(
                   [now],[older],[anchor-lag],[anchor],lag
                )[0]
                inputs[j]=(base[site].clone(),state.memory[site].clone(),
                           source_obs[site].clone(),query)
    if any(x is None for x in inputs):
        raise RuntimeError("unread genuine source past query anchors")
    # Now, and only now, look up the already-recorded PAST source observation.
    # It was in the memory write history, not injected into the head inputs.
    labels=[]
    for site,anchor,lag,older,now in queries:
        if older>=now or older!=rows[site][anchor-lag]:
            raise RuntimeError("past event source leakage")
        target=data["values"][site,older,0]
        if not data["valid"][site,older,0].item():
            raise RuntimeError("historical source measurement not actually observed")
        labels.append(target)
    return tuple(torch.stack([x[k] for x in inputs]) for k in range(4)),torch.stack(labels)

def preregister(source):
    if PLAN.exists() or OUT.exists():
        raise RuntimeError("source genuine query plan or original fit already exists")
    data,fit,dev=source_and_times(source)
    original=frozen(source)
    receipt=verify_originals(original)
    train,development=timestamps(fit,dev)
    n=len(train)
    plans={}
    for seed in SEEDS:
        rng=random.Random(seed+6006)
        plans[str(seed)]=[rng.randrange(n) for _ in range(TRAIN_STEPS*BATCH)]
    p=dict(study_id="PAR-MEM-006-GENUINE-HISTORIC-QUERY",
         original_publisher_SHA256=SHA256,
         pretrained_genuine_model_receipt_sha256=checksum(SOURCE_ROOT/"TRAIN_RECEIPT.json"),
         genuine_train_history_source_sha256=original["train_source_clock_SHA256"],
         known_exposed_DEV_source_sha256=original["exposed_DEV_source_clock_SHA256"],
         true_train_queries=len(train),true_known_DEV_queries=len(development),
         fixed_train_source_query_clock_SHA256=digest(train),
         fixed_known_DEV_source_query_clock_SHA256=digest(development),
         original_query_clock_records_train=train,
         original_query_clock_records_DEV=development,
         exact_seed_source_sample_order=plans,
         sample_hash_by_seed={key:digest(value) for key,value in plans.items()},
         no_neural_query_head_training_preceded_commit=True,
         full_mathematical_training_FLOP_equal_arm_certified=False,
         actual_source_observation_targets_excluded_from_query_metadata=True)
    PLAN.write_text(json.dumps(p,indent=2,sort_keys=True)+"\n")
    print("PAR_MEM_006_ORIGINAL_PLAN_BEFORE_FIT="+json.dumps({
        "training_queries":len(train),"already_exposed_DEV_queries":len(development),
        "seed_schedule_hash":p["sample_hash_by_seed"],
        "original_models":5,"head_models_trained":0
    },sort_keys=True),flush=True)

def fixed_plan(source):
    if not PLAN.exists():raise RuntimeError("original query times must be frozen before FIT")
    original=frozen(source)
    receipt=verify_originals(original)
    data,fit,dev=source_and_times(source)
    t,v=timestamps(fit,dev)
    p=json.loads(PLAN.read_text())
    if (p["study_id"]!="PAR-MEM-006-GENUINE-HISTORIC-QUERY"
       or p["original_publisher_SHA256"]!=SHA256
       or p["pretrained_genuine_model_receipt_sha256"]!=checksum(SOURCE_ROOT/"TRAIN_RECEIPT.json")
       or p["genuine_train_history_source_sha256"]!=original["train_source_clock_SHA256"]
       or p["fixed_train_source_query_clock_SHA256"]!=digest(t)
       or p["fixed_known_DEV_source_query_clock_SHA256"]!=digest(v)
       or p["original_query_clock_records_train"]!=[list(x) for x in t]
       or p["original_query_clock_records_DEV"]!=[list(x) for x in v]):
        raise RuntimeError("actual source query locations changed after plan freeze")
    for seed in SEEDS:
        rng=random.Random(seed+6006)
        a=[rng.randrange(len(t)) for _ in range(TRAIN_STEPS*BATCH)]
        if (p["exact_seed_source_sample_order"][str(seed)]!=a
           or p["sample_hash_by_seed"][str(seed)]!=digest(a)):
            raise RuntimeError("original genuine query optimizer schedule changed")
    return p,(data,fit,dev,t,v),receipt

def fit(source):
    if OUT.exists():raise RuntimeError("trained source query head original directory exists")
    p,(data,fit,_,train,_),receipt=fixed_plan(source)
    OUT.mkdir(parents=True,exist_ok=False)
    records=[]
    for seed in SEEDS:
        features,target=features_from_timed_stream(seed,fit,train,receipt,data)
        chosen=p["exact_seed_source_sample_order"][str(seed)]
        for arm in ARMS:
            head=model_pair(seed,arm)
            optimizer=torch.optim.AdamW(head.parameters(),lr=LR,
                                         weight_decay=.001,foreach=False,fused=False)
            first=[q.detach().clone() for q in head.parameters()]
            loss_trace=[]
            start=time.monotonic()
            for update in range(TRAIN_STEPS):
                ids=torch.tensor(chosen[update*BATCH:(update+1)*BATCH],dtype=torch.long)
                X=[z[ids] for z in features]
                y=target[ids]
                optimizer.zero_grad(set_to_none=True)
                pred=head(*X,associative=(arm=="associative_read"))
                loss=F.smooth_l1_loss(pred,y)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("true historic source query loss nonfinite")
                loss.backward()
                if any(q.grad is None or not torch.isfinite(q.grad).all().item()
                       for q in head.parameters()):
                    raise RuntimeError("trained query head disconnected or dummy weights")
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.0)
                optimizer.step()
                loss_trace.append(float(loss.detach()))
            if not any(not torch.equal(a,b) for a,b in zip(first,head.parameters())):
                raise RuntimeError("trained source query head weights unchanged")
            folder=OUT/f"seed-{seed}-{arm}.pt"
            torch.save(dict(seed=seed,arm=arm,model_state=head.state_dict(),
               AdamW=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
               python_rng=random.getstate(),trained_head_parameters=3913,
               full_deployed_active_parameters=100552,
               pretrained_original_checkpoint_SHA256=next(
                   q["original_checkpoint_SHA256"] for q in receipt["original_checkpoints"]
                   if q["seed"]==seed),
               source_sha256=SHA256,
               original_source_plan_SHA256=digest(p["original_query_clock_records_train"]),
               original_sample_schedule_SHA256=p["sample_hash_by_seed"][str(seed)],
               code_SHA256=checksum(ROOT/"scripts/parmem006_source_history_query.py"),
               training_steps=TRAIN_STEPS,batch=BATCH,
               loss_trace=loss_trace,no_independent_qualification=True),folder)
            row=dict(seed=seed,arm=arm,trained_query_head_SHA256=checksum(folder),
                     optimizer_steps=TRAIN_STEPS,examples_seen=TRAIN_STEPS*BATCH,
                     start_training_loss=loss_trace[0],last_training_loss=loss_trace[-1],
                     mean_last_16_training_loss=sum(loss_trace[-16:])/16,
                     CPU_elapsed_seconds_outcome_only=time.monotonic()-start)
            records.append(row)
            print("PAR_MEM_006_ORIGINAL_REAL_TRAINED="+json.dumps(row,sort_keys=True),flush=True)
    r=dict(study_id="PAR-MEM-006-GENUINE-HISTORIC-QUERY",
         original_model_count=len(records),genuine_source_SHA256=SHA256,
         physical_training_query_count=len(train),
         source_data_previous_years_protected=False,
         trained_frozen_backbone_retrieval_head_arms=list(ARMS),
         comparator_full_training_FLOP_parity_QUALIFIED=False,
         original_models_FROZEN_BEFORE_DEV=True,
         original_training_records=records)
    (OUT/"TRAIN_RECEIPT.json").write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")

def verify_heads(p,receipt):
    r=json.loads((OUT/"TRAIN_RECEIPT.json").read_text())
    if len(r["original_training_records"])!=10:
        raise RuntimeError("10 real trained paired query heads original freeze missing")
    for item in r["original_training_records"]:
        saved=OUT/f"seed-{item['seed']}-{item['arm']}.pt"
        if checksum(saved)!=item["trained_query_head_SHA256"]:
            raise RuntimeError("genuine original query head SHA changed")
        st=torch.load(saved,map_location="cpu",weights_only=False)
        if (st["seed"]!=item["seed"] or st["arm"]!=item["arm"]
            or st["source_sha256"]!=SHA256
            or st["code_SHA256"]!=checksum(ROOT/"scripts/parmem006_source_history_query.py")
            or st["original_source_plan_SHA256"]!=digest(p["original_query_clock_records_train"])
            or st["original_sample_schedule_SHA256"]!=p["sample_hash_by_seed"][str(item["seed"])]
            or st["training_steps"]!=TRAIN_STEPS or st["full_deployed_active_parameters"]!=100552):
            raise RuntimeError("original source query checkpoint or training provenance invalid")
    return r

def evaluate(source,*,replay=False):
    p,(data,_,dev,_,questions),base_receipt=fixed_plan(source)
    r=verify_heads(p,base_receipt)
    if (OUT/"DEV_RESULTS.json").exists()!=replay:
        raise RuntimeError("historical source result may be created once and only replayed")
    reports=[];original_tensors={}
    for seed in SEEDS:
        features,target=features_from_timed_stream(seed,dev,questions,base_receipt,data)
        qmeta=features[-1]
        lag_tensor=torch.tensor([int(q[2]) for q in questions])
        for arm in ARMS:
            row=next(x for x in r["original_training_records"]
                     if x["seed"]==seed and x["arm"]==arm)
            saved=torch.load(OUT/f"seed-{seed}-{arm}.pt",map_location="cpu",weights_only=False)
            head=model_pair(seed,arm)
            head.load_state_dict(saved["model_state"],strict=True)
            head.eval()
            with torch.no_grad():
                pred=head(*features,associative=arm=="associative_read")
                # Legitimate source *current* PM25, NOT the exact past target.
                current=features[2][:,0]
                true=target
                band={}
                for lag in LAGS:
                    ids=lag_tensor==lag
                    band[str(lag)]=dict(
                        genuine_source_measured_queries=int(ids.sum()),
                        historical_normalized_PM25_MAE=float((pred[ids]-true[ids]).abs().mean()),
                        known_now_source_PM25_persistence_MAE=float((current[ids]-true[ids]).abs().mean()),
                        observed_true_history_oracle_MAE=0.,
                        source_training_oracle_can_query_truth_from_external_full_record=True
                    )
            arr=dict(prediction=pred.detach(),target=true.detach(),current=current.detach())
            file=OUT/f"seed-{seed}-{arm}-DEV_original.pt"
            if replay:
                old=torch.load(file,map_location="cpu",weights_only=True)
                for k in arr:
                    torch.testing.assert_close(old[k],arr[k],rtol=1e-5,atol=1e-6)
            else:torch.save(arr,file)
            reports.append(dict(seed=seed,arm=arm,age_lag_evaluation=band,
                                source_checkpoint_SHA256=row["trained_query_head_SHA256"]))
    record=dict(study_id="PAR-MEM-006-GENUINE-HISTORIC-QUERY",
       original_publisher_SHA256=SHA256,
       trained_query_heads=10,
       all_genuine_measurement_targets=True,
       no_future_source_labels_used_in_queries=True,
       prev_exposed_DEV_genuine_observation=True,
       matched_train_budget=True,equal_trainable_and_deployed_params=True,
       complete_training_FLOP_parity_not_verified=True,
       independent_architecture_superiority_QUALIFIED=False,
       original_source_records=reports)
    output=OUT/"DEV_RESULTS.json"
    if replay:
        old=json.loads(output.read_text())
        def near(a,b):
            if isinstance(a,dict):
                if set(a)!=set(b):raise RuntimeError("source score schema mismatch")
                for k in a: near(a[k],b[k])
            elif isinstance(a,list):
                if len(a)!=len(b):raise RuntimeError("source score history length mismatch")
                for x,y in zip(a,b):near(x,y)
            elif isinstance(a,float):
                if abs(a-b)>1e-5*max(1,abs(a)):raise RuntimeError("historical replay mismatch")
            elif a!=b:raise RuntimeError("original source query identity mismatch")
        near(old,record)
        print("PAR_MEM_006_INDEPENDENT_SOURCE_HISTORY_REPLAY_PASS")
    else:
        output.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
        print("PAR_MEM_006_REAL_SOURCE_DEV_10_ORIGINAL_QUERY_HEADS="+json.dumps({
           "training_arms":list(ARMS),
           "previously_exposed_DEV_queries":len(questions),
           "results":reports},sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("plan","fit","eval","replay"),required=True)
    p.add_argument("--source",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    if args.mode=="plan":preregister(args.source)
    elif args.mode=="fit":fit(args.source)
    else:evaluate(args.source,replay=args.mode=="replay")
