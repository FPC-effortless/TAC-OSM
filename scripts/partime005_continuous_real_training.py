"""PAR-TIME-005 genuine long-stream TAC training and exact replay.

SINGLE TIME-AWARE TRAINED ARM ONLY. All source clocks and measurements are
publisher-observed. 12 station-specific chronological streams carry state
across 24 consecutive 32-event truncated-BPTT optimizer blocks (768 genuine
event observations per station). The persistent state is NOT reset between
updates; gradients ARE detached at chunk boundaries and weights change.
A new development stream is evaluated only AFTER all original 5 checkpoints
are SHA frozen. Previously-exposed real 2014 May-Jun DEV is NOT untouched TEST.
No future labels select source cases or enter prediction inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import torch

from scripts.partime002_real_source import (
    SHA256, SITES, FIT_START, FIT_STOP, EXPOSED_START, EXPOSED_STOP,
    hour, read_source, score,
)
from tac_osm.variants.parallel_time_age_world import TimeAwareParallelResidualWorld

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/parallel-six/PAR-TIME-005-PROTOCOL.json"
PLAN=ROOT/"research/parallel-six/PAR-TIME-005-FROZEN-PLAN.json"
OUT=ROOT/"research/replay-bundles/PAR-TIME-005"
SEEDS=(271028,271029,271030,271031,271032)
UPDATES=24
BLOCK=32
N=UPDATES*BLOCK
B=len(SITES)
LR=0.0003
BANDS=(("32-63",32,64),("64-127",64,128),("128-255",128,256),
       ("256-511",256,512),("512+",512,1_000_000))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def model_for(seed):
    torch.manual_seed(seed)
    model=TimeAwareParallelResidualWorld()
    parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameters!=96639 or any(p.dtype!=torch.float32 for p in model.parameters()):
        raise RuntimeError("96,639 active float32 parameter preregistration changed")
    return model

def source_and_times(zip_path):
    c=json.loads(REG.read_text())
    if (c["study_id"]!="PAR-TIME-005-CONTINUOUS-SOURCE-STREAMS"
        or c["status"]!="SINGLE_TIME_AWARE_ARM_DEV_NO_MATCHED_NEURAL_COMPARISON"
        or c["source_zip_SHA256"]!=SHA256
        or c["training_seed_plan"]!=list(SEEDS)
        or c["fit_updates_per_seed"]!=UPDATES
        or c["tbptt_events_per_update"]!=BLOCK
        or c["batch_parallel_stations"]!=B
        or c["active_learned_parameters"]!=96639
        or c["learning_rate"]!=LR):
        raise RuntimeError("immutable source/model optimizer protocol drift")
    data=read_source(zip_path)
    fit=[]
    dev=[]
    for site,original in enumerate(data["events"]):
        train=[t for t in original if hour(FIT_START)<=t<hour(FIT_STOP)-1]
        development=[t for t in original
                     if hour(EXPOSED_START)<=t<hour(EXPOSED_STOP)-1]
        if len(train)<N or len(development)<576:
            raise RuntimeError(f"insufficient true observed site history at {site}: {len(train)} train {len(development)} DEV")
        fit.append(train[:N])
        dev.append(development)
    if max(x[-1] for x in fit)+1>=min(x[0] for x in dev):
        raise RuntimeError("genuine source training target overlaps development input")
    return data,fit,dev

def generate_plan(zip_path):
    if PLAN.exists() or OUT.exists():
        raise RuntimeError("genuine source plan or checkpoints already exist")
    _,fit,dev=source_and_times(zip_path)
    original=dict(
       study_id="PAR-TIME-005-CONTINUOUS-SOURCE-STREAMS",
       original_publisher_source_SHA256=SHA256,
       seed_list=list(SEEDS),
       chronological_train_hour_indices_by_station=fit,
       chronological_DEV_hour_indices_by_station=dev,
       train_source_clock_SHA256=digest(fit),
       exposed_DEV_source_clock_SHA256=digest(dev),
       training_steps_per_seed=UPDATES,
       observed_events_per_train_station=N,
       train_batch_stations=B,
       optimizer_updates_not_yet_run=True,
       target_never_used_to_select_training_timestamp=True,
       development_previously_exposed_and_NOT_new_qualification=True
    )
    PLAN.write_text(json.dumps(original,indent=2,sort_keys=True)+"\n")
    print("PAR_TIME_005_FROZEN_PLAN_BEFORE_TRAIN="+json.dumps({
        "source_SHA":SHA256,
        "per_site_train_events":[len(x) for x in fit],
        "per_site_DEV_events":[len(x) for x in dev],
        "train_timing_sha":original["train_source_clock_SHA256"],
        "DEV_timing_sha":original["exposed_DEV_source_clock_SHA256"],
        "trained_model_count":0,
    },sort_keys=True),flush=True)

def frozen(zip_path):
    if not PLAN.exists():raise RuntimeError("source times must be precommitted before any fit")
    _,fit,dev=source_and_times(zip_path)
    plan=json.loads(PLAN.read_text())
    if (plan["study_id"]!="PAR-TIME-005-CONTINUOUS-SOURCE-STREAMS"
        or plan["original_publisher_source_SHA256"]!=SHA256
        or plan["seed_list"]!=list(SEEDS)
        or plan["chronological_train_hour_indices_by_station"]!=fit
        or plan["chronological_DEV_hour_indices_by_station"]!=dev
        or plan["train_source_clock_SHA256"]!=digest(fit)
        or plan["exposed_DEV_source_clock_SHA256"]!=digest(dev)
        or plan["training_steps_per_seed"]!=UPDATES
        or plan["observed_events_per_train_station"]!=N):
        raise RuntimeError("original source chronological schedule changed after prefit")
    return plan

def observed_source_step(data,clocks,previous):
    """True source event at each independent station. No numeric future reads.

    clocks: 12 actual source observed timestamps; previous is previous
    timestamp of same site's last admitted observation or None on cold start.
    """
    time_ix=torch.tensor(clocks,dtype=torch.long)
    site=torch.arange(B,dtype=torch.long)
    values=data["values"][site,time_ix]
    valid=data["valid"][site,time_ix]
    obs=torch.cat((values,valid.float().mean(-1,keepdim=True)),dim=-1)
    known_hour=torch.sin((time_ix%24).float()*2*torch.pi/24).unsqueeze(-1)
    covar=torch.cat((valid.float(),known_hour),dim=-1)
    dt=torch.zeros(B,dtype=torch.float32) if previous is None else (
         time_ix-torch.tensor(previous,dtype=torch.long)).float()
    if (dt<0).any().item() or (previous is not None and (dt<=0).any().item()):
        raise RuntimeError("backward/duplicate station source timestamp")
    admit=torch.ones(B,dtype=torch.bool)  # SAME source-authorized criterion at all stations
    return obs,covar,dt,admit

def true_future(data,clocks):
    # Must be called strictly AFTER the model has produced a prediction for
    # these source inputs in both FIT and DEV. Independent true t+1 sensors.
    next_ix=torch.tensor(clocks,dtype=torch.long)+1
    site=torch.arange(B,dtype=torch.long)
    y=data["values"][site,next_ix]
    valid=data["valid"][site,next_ix]
    target=torch.cat((y,valid.float().mean(-1,keepdim=True)),dim=-1)
    return target,valid

def detached(state):
    # Numeric persistent state survives, AD graph is intentionally truncated
    # at the physical 32-event source update boundary.
    return replace(state,
       fast_sum=state.fast_sum.detach(),
       memory=state.memory.detach(),
       elapsed_clock=state.elapsed_clock.detach(),
       last_write_clock=state.last_write_clock.detach(),
       write_seen=state.write_seen.detach())

def fit(zip_path):
    if OUT.exists():raise RuntimeError("original trained neural checkpoint directory exists; never overwrite")
    p=frozen(zip_path)
    data,fit_clocks,_=source_and_times(zip_path)
    OUT.mkdir(parents=True,exist_ok=False)
    results=[]
    for seed in SEEDS:
        random.seed(seed)
        model=model_for(seed)
        optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.001,
                                   foreach=False,fused=False)
        initial_weights=[v.detach().clone() for v in model.parameters()]
        model.train()
        state=model.init_stream(B)
        before=None
        losses=[]
        start=time.monotonic()
        for update in range(UPDATES):
            optimizer.zero_grad(set_to_none=True)
            terms=[]
            for offset in range(BLOCK):
                index=update*BLOCK+offset
                clocks=[fit_clocks[s][index] for s in range(B)]
                observed,action,delta,admit=observed_source_step(data,clocks,before)
                prediction,state=model.stream_step(state,observed,action,
                               elapsed_dt=delta,write_mask=admit)
                true,valid=true_future(data,clocks)
                terms.append(score(prediction,true,valid))
                before=clocks
            loss=torch.stack(terms).mean()
            if not torch.isfinite(loss).item():
                raise RuntimeError("true source training loss diverged")
            loss.backward()
            if any(v.grad is None or not v.grad.isfinite().all().item()
                   for v in model.parameters()):
                raise RuntimeError("nonfinite/disconnected recurrent source gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(float(loss.detach()))
            # DO NOT reset state! The same station memories continue into the
            # next optimizer update with truncated backprop after this step.
            state=detached(state)
            if state.count!=(update+1)*BLOCK:
                raise RuntimeError("continuous 12-station state unexpectedly reset")
        if state.count!=N or not state.write_seen.all().item():
            raise RuntimeError("continuous source state not present at fit end")
        if not any(not torch.equal(a,b) for a,b in zip(initial_weights,model.parameters())):
            raise RuntimeError("source training made no parameter updates")
        target=OUT/f"seed-{seed}-original-continuous.pt"
        torch.save(dict(
            study_id="PAR-TIME-005-CONTINUOUS-SOURCE-STREAMS",
            seed=seed,model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            pytorch_rng=torch.get_rng_state(),python_rng=random.getstate(),
            final_persistent_stream=dict(
                fast_sum=state.fast_sum.detach(),memory=state.memory.detach(),
                elapsed_clock=state.elapsed_clock.detach(),
                last_write_clock=state.last_write_clock.detach(),
                write_seen=state.write_seen.detach(),count=state.count,
            ),
            original_train_clock_SHA256=p["train_source_clock_SHA256"],
            previously_exposed_DEV_clock_SHA256=p["exposed_DEV_source_clock_SHA256"],
            publisher_source_sha256=SHA256,
            model_source_sha256=sha(ROOT/"src/tac_osm/variants/parallel_time_age_world.py"),
            adapter_source_sha256=sha(ROOT/"scripts/partime002_real_source.py"),
            runner_source_sha256=sha(ROOT/"scripts/partime005_continuous_real_training.py"),
            optimizer_update_count=UPDATES,
            continuous_genuine_observed_events_per_station=N,
            retained_state_not_reset_between_fit_updates=True,
            bptt_truncated_every_32_events=True,
            training_loss_trace=losses,
            no_untouched_source_test_evaluated=True,
        ),target)
        row=dict(seed=seed,original_checkpoint_SHA256=sha(target),
                 optimizer_updates=UPDATES,continuous_observed_events_per_station=N,
                 observed_source_stations=B,trained_parameters=96639,
                 observed_training_site_steps=N*B,
                 first_train_loss=losses[0],last_train_loss=losses[-1],
                 last_5_mean=sum(losses[-5:])/5,
                 final_state_count=state.count,
                 matrix_mean_abs=float(state.memory.detach().abs().mean()),
                 CPU_training_elapsed_seconds_outcome_only=time.monotonic()-start)
        results.append(row)
        print("PAR_TIME_005_REAL_CONTINUOUS_TRAINED="+json.dumps(row,sort_keys=True),flush=True)
    receipt=dict(study_id="PAR-TIME-005-CONTINUOUS-SOURCE-STREAMS",
         source_sha256=SHA256,
         model_count=len(results),
         source_time_plan_sha256=p["train_source_clock_SHA256"],
         all_original_checkpoints_FROZEN_BEFORE_DEV=True,
         all_genuine_training_runs_persisted_state_across_optimizer_steps=True,
         trained_real_source_model_comparison_is_single_arm=True,
         matched_full_TRAIN_FLOP_parity_against_competitor=False,
         previously_exposed_DEV_scored=False,
         original_checkpoints=results)
    (OUT/"TRAIN_RECEIPT.json").write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("PAR_TIME_005_ALL_FIVE_TRAINED_FROZEN_BEFORE_DEV",flush=True)

def verify_originals(plan):
    receipt=json.loads((OUT/"TRAIN_RECEIPT.json").read_text())
    if (receipt["model_count"]!=5 or
        set(x["seed"] for x in receipt["original_checkpoints"])!=set(SEEDS)):
        raise RuntimeError("original source trained seed family incomplete")
    for row in receipt["original_checkpoints"]:
        ck=OUT/f"seed-{row['seed']}-original-continuous.pt"
        if sha(ck)!=row["original_checkpoint_SHA256"]:
            raise RuntimeError("original trained checkpoint was modified")
        cp=torch.load(ck,map_location="cpu",weights_only=False)
        if (cp["seed"]!=row["seed"]
            or cp["publisher_source_sha256"]!=SHA256
            or cp["original_train_clock_SHA256"]!=plan["train_source_clock_SHA256"]
            or cp["previously_exposed_DEV_clock_SHA256"]!=plan["exposed_DEV_source_clock_SHA256"]
            or cp["optimizer_update_count"]!=UPDATES
            or cp["continuous_genuine_observed_events_per_station"]!=N
            or cp["final_persistent_stream"]["count"]!=N
            or cp["model_source_sha256"]!=sha(ROOT/"src/tac_osm/variants/parallel_time_age_world.py")
            or cp["adapter_source_sha256"]!=sha(ROOT/"scripts/partime002_real_source.py")
            or cp["runner_source_sha256"]!=sha(ROOT/"scripts/partime005_continuous_real_training.py")):
            raise RuntimeError("original source frozen model/state/code provenance failed")
    return receipt

def fresh_dev_predictions(model,data,dev_times):
    model.eval()
    state=model.init_stream(B)
    previous=None
    per_band={name:dict(pred_abs=0.,pers_abs=0.,valid=0,
                        pm_pred_abs=0.,pm_pers_abs=0.,pm_valid=0,
                        observed_station_event_steps=0)
              for name,_,_ in BANDS}
    max_events=min(len(x) for x in dev_times)
    signatures=[]
    with torch.no_grad():
        for index in range(max_events):
            clocks=[dev_times[s][index] for s in range(B)]
            x,a,delta,mask=observed_source_step(data,clocks,previous)
            y,state=model.stream_step(state,x,a,elapsed_dt=delta,write_mask=mask)
            truth,valid=true_future(data,clocks)
            previous=clocks
            # Save every predicted real next-hour 14D tensor as replay output,
            # not just aggregated errors; only time>=32 is qualified for bin.
            signatures.append(y.detach().clone())
            current=[name for name,low,high in BANDS if low<=index<high]
            if not current:continue
            name=current[0]
            row=per_band[name]
            present=valid.float()
            row["pred_abs"]+=float(((y[:,:13]-truth[:,:13]).abs()*present).sum())
            row["pers_abs"]+=float(((x[:,:13]-truth[:,:13]).abs()*present).sum())
            row["valid"]+=int(present.sum())
            row["pm_pred_abs"]+=float(((y[:,0]-truth[:,0]).abs()*present[:,0]).sum())
            row["pm_pers_abs"]+=float(((x[:,0]-truth[:,0]).abs()*present[:,0]).sum())
            row["pm_valid"]+=int(present[:,0].sum())
            row["observed_station_event_steps"]+=B
    if state.count!=max_events or not state.write_seen.all().item():
        raise RuntimeError("source evaluation did not carry independent station state")
    scores={}
    for name,entry in per_band.items():
        if entry["valid"]<40 or entry["pm_valid"]<3:
            # An absent high-age band is not a zero score or a new skill.
            scores[name]=dict(eligible=False,source_events=entry["observed_station_event_steps"])
            continue
        scores[name]=dict(eligible=True,source_events=entry["observed_station_event_steps"],
              true_observed_source_channels=entry["valid"],
              true_PM25_labels=entry["pm_valid"],
              normalized_MAE=entry["pred_abs"]/entry["valid"],
              normalized_persistence_MAE=entry["pers_abs"]/entry["valid"],
              normalized_PM25_MAE=entry["pm_pred_abs"]/entry["pm_valid"],
              normalized_PM25_persistence_MAE=entry["pm_pers_abs"]/entry["pm_valid"])
    return torch.stack(signatures),scores,dict(events=max_events,
          final_count=state.count,real_source_total_clock_mean=float(state.elapsed_clock.mean()),
          matrix_mean_abs=float(state.memory.abs().mean()))

def evaluate(zip_path,*,replay=False):
    p=frozen(zip_path)
    receipt=verify_originals(p)
    dev_path=OUT/"DEV_RESULTS.json"
    if replay and not dev_path.exists():raise RuntimeError("original source DEV receipt missing")
    if not replay and dev_path.exists():raise RuntimeError("source DEV already examined; never rescore")
    data,_,dev=source_and_times(zip_path)
    reports=[]
    for item in receipt["original_checkpoints"]:
        seed=item["seed"]
        saved=torch.load(OUT/f"seed-{seed}-original-continuous.pt",
                         map_location="cpu",weights_only=False)
        model=model_for(seed)
        model.load_state_dict(saved["model_state"],strict=True)
        prediction,bands,state=fresh_dev_predictions(model,data,dev)
        array_path=OUT/f"seed-{seed}-DEV-original-predictions.pt"
        if replay:
            old=torch.load(array_path,map_location="cpu",weights_only=True)
            torch.testing.assert_close(prediction,old,rtol=1e-5,atol=1e-6)
        else:
            torch.save(prediction,array_path)
        reports.append(dict(seed=seed,checkpoint_SHA256=item["original_checkpoint_SHA256"],
                            source_development_continuous_state=state,
                            age_band_metrics=bands))
        print("PAR_TIME_005_ORIGINAL_REAL_DEV_SEED="+json.dumps(reports[-1],sort_keys=True),flush=True)
    result=dict(study_id="PAR-TIME-005-CONTINUOUS-SOURCE-STREAMS",
       source_SHA256=SHA256,trained_models=5,
       original_train_source_SHA256=p["train_source_clock_SHA256"],
       original_DEV_source_time_SHA256=p["exposed_DEV_source_clock_SHA256"],
       trained_continuous_768_source_events_per_station=True,
       original_model_has_persistent_stream_state=True,
       data_period_previously_exposed_DEV=True,
       qualification="UNQUALIFIED",
       matched_learned_architecture_comparison=False,
       trained_model_no_long_memory_superiority_claim=True,
       years_2015_2017_numerical_never_parsed=True,
       original_records=reports)
    if replay:
        original=json.loads(dev_path.read_text())
        def near(a,b,k="root"):
            if isinstance(a,dict):
                if set(a)!=set(b):raise RuntimeError("original source result key changed: "+k)
                for key in a:near(a[key],b[key],k+"."+str(key))
            elif isinstance(a,list):
                if len(a)!=len(b):raise RuntimeError("real source model seed length changed: "+k)
                for i,(x,y) in enumerate(zip(a,b)):near(x,y,k+"."+str(i))
            elif isinstance(a,float):
                if abs(a-b)>1e-6*max(1.,abs(a)):raise RuntimeError("source replay metric drift: "+k)
            elif a!=b:
                raise RuntimeError("original source result identity changed: "+k)
        near(original,result)
        print("PAR_TIME_005_INDEPENDENT_ORIGINAL_LONG_SOURCE_REPLAY_PASS",flush=True)
    else:
        dev_path.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("plan","fit","eval","replay"),required=True)
    p.add_argument("--source",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    if args.mode=="plan":generate_plan(args.source)
    elif args.mode=="fit":fit(args.source)
    else:evaluate(args.source,replay=(args.mode=="replay"))

if __name__=="__main__":
    main()
