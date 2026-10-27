"""PAR-TIME-004: READ-ONLY true-source FIT-period temporal and memory lesions.

No optimizer, no writes to original checkpoint, no previously exposed DEV
numerical evaluation, no independent TEST access and no architecture head-to-head.

Original 5 PAR-TIME-003 trained checkpoint hashes MUST match original receipts.
Source case selection depends ONLY on genuinely recorded prior event clocks.
The model was trained on 12-event histories and resets state each window.
A continued stream is software-capable; trained long-window retention and
cross-process service persistence are NOT demonstrated here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import random

import torch

from scripts.partime002_real_source import (
    SHA256, read_source, chronology, batch_from_cases,
)
from scripts.partime003_single_model_fit import (
    OUT as ORIGINAL_OUT, SEEDS, modelspec, file_hash, freeze,
    check_original_weights,
)
from tac_osm.variants.parallel_time_age_world import TimeAwareStreamState

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/parallel-six/PAR-TIME-004-PROTOCOL.json"
OUT=ROOT/"research/parallel-six/PAR-TIME-004-RESULTS.json"
COHORT_SEED=271027
SPLIT=6
FIXED_LENGTH=12
ARMS={
    "zero_position_columns":[0,1,2],
    "zero_elapsed_delta_column":[3],
    "zero_elapsed_since_origin_column":[4],
    "zero_last_write_age_and_seen_columns":[5,6],
    "zero_all_temporal_columns":[0,1,2,3,4,5,6],
}

def sha_rows(rows):
    return hashlib.sha256(
        json.dumps(rows,sort_keys=True,separators=(",",":")).encode()
    ).hexdigest()

def cohort(data,registered_train):
    regular=[]
    irregular=[]
    for case in registered_train:
        site,ix,_=case
        clocks=data["events"][site][ix-FIXED_LENGTH+1:ix+1]
        if len(clocks)!=FIXED_LENGTH:
            raise RuntimeError("registered original source case has no history")
        largest=max(y-x for x,y in zip(clocks,clocks[1:]))
        (irregular if largest>1 else regular).append(case)
    if min(len(regular),len(irregular))<32:
        raise RuntimeError("insufficient source-authorized regular/irregular histories")
    g=random.Random(COHORT_SEED)
    selected=g.sample(irregular,32)+g.sample(regular,32)
    return selected,dict(regular=len(regular),irregular=len(irregular),
                         selected_regular=32,selected_irregular=32,
                         selected_cases_sha256=sha_rows(selected))

def assert_contract(cfg):
    if (cfg["study_id"]!="PAR-TIME-004-FROZEN-FIT-ONLY-TEMPORAL-AND-MEMORY-LESIONS"
        or cfg["source_archive_sha256"]!=SHA256
        or cfg["cohort_selection"]["seed"]!=COHORT_SEED
        or cfg["cohort_selection"]["fixed_case_count"]!=64
        or cfg["original_checkpoint_seeds"]!=list(SEEDS)
        or cfg["optimizer_steps"]!=0 or cfg["checkpoint_mutations"]!=0
        or cfg["forward_only"] is not True
        or cfg["timing_columns"]!={
           "position":[0,1,2], "elapsed_delta":[3],
           "since_sequence_origin":[4],
           "most_recent_previous_admitted_write_age_and_seen":[5,6],
           "all_time_features":[0,1,2,3,4,5,6],
        }):
        raise RuntimeError("precommitted real FIT-only attribution contract changed")
    if OUT.exists():
        raise RuntimeError("original real-source attribution receipt already exists; no overwrite")

def predict_batch(model,batch, *,return_trace=False):
    observation,covariates,delta,admit,*_=batch
    return model(observation,covariates,elapsed_dt=delta,
                 write_mask=admit,return_trace=return_trace)

def saved_roundtrip(state:TimeAwareStreamState):
    b=io.BytesIO()
    payload=asdict(state)
    torch.save(payload,b)
    b.seek(0)
    restored=torch.load(b,map_location="cpu",weights_only=True)
    if set(restored)!=set(payload):
        raise RuntimeError("typed stream state payload schema drift")
    for k in payload:
        if isinstance(payload[k],torch.Tensor):
            if not torch.equal(payload[k],restored[k]):
                raise RuntimeError("source stream state serialization mismatch: "+k)
        elif payload[k]!=restored[k]:
            raise RuntimeError("stream index was lost during state serialization")
    return TimeAwareStreamState(**restored)

def sequential_diagnostic(model,batch,*,remove_matrix_between_steps=False,
                          do_roundtrip=False):
    observation,covariates,delta,admit,*_=batch
    B,T,_=observation.shape
    state=model.init_stream(B)
    preds=[]
    memory_norm=[]
    for t in range(T):
        if remove_matrix_between_steps:
            state=replace(state,memory=torch.zeros_like(state.memory))
        if do_roundtrip and t==SPLIT:
            state=saved_roundtrip(state)
        prediction,state=model.stream_step(
            state,observation[:,t],covariates[:,t],
            elapsed_dt=delta[:,t],write_mask=admit[:,t])
        preds.append(prediction)
        memory_norm.append(float(state.memory.abs().mean()))
    if (state.count!=T or not torch.equal(
        state.elapsed_clock,delta.sum(1))):
        raise RuntimeError("causal recurrent counter/count corrupted")
    if not state.write_seen.all().item():
        raise RuntimeError("registered admitted source observations dropped")
    return torch.stack(preds,dim=1),state,memory_norm

def metrics(pred,batch):
    *_,target,valid=batch
    mask=valid[:,:13]
    residual=(pred[:,:13]-target[:,:13]).abs()
    pm=mask[:,0]
    n=int(mask.sum())
    n_pm=int(pm.sum())
    if n<40 or n_pm<2:
        raise RuntimeError("actual FIT source lacks measured future targets")
    return dict(observed_channels=n,observed_pm25=n_pm,
                absolute_channel_error_sum=float((residual*mask).sum()),
                absolute_pm25_error_sum=float(residual[:,0][pm].sum()))

def source_probe(model,batch):
    with torch.no_grad():
        original,trace=predict_batch(model,batch,return_trace=True)
        inp_original=original[:,-1]
        memory=trace["memory"]
        # Original source all-true write mask: last-write-age and per-step
        # elapsed duration are identical by construction; model cannot
        # separately identify these two coordinates on THIS fitted source.
        time=trace["time_features"]
        time_dupe=float((time[:,:,5]-time[:,:,3]).abs().max())
        if time_dupe>1e-5:
            raise RuntimeError("all-write source age and dt identity violated")
        stream,full_state,norms=sequential_diagnostic(model,batch)
        torch.testing.assert_close(stream,original,atol=3e-4,rtol=7e-4)
        saved,restored,_=sequential_diagnostic(model,batch,do_roundtrip=True)
        torch.testing.assert_close(saved,stream,atol=0,rtol=0)
        zeroed,_,_=sequential_diagnostic(
            model,batch,remove_matrix_between_steps=True)
        original_weight=model.core.time_projection.weight.detach().clone()
        lesions={}
        try:
            for name,cols in ARMS.items():
                # Only in-RAM, transient, with original original SHA-frozen
                # trained weights immediately restored below.
                model.core.time_projection.weight[:,cols]=0
                lesioned=predict_batch(model,batch)[:,-1]
                lesions[name]=lesioned.clone()
                model.core.time_projection.weight.copy_(original_weight)
        finally:
            model.core.time_projection.weight.copy_(original_weight)
        if not torch.equal(original_weight,model.core.time_projection.weight):
            raise RuntimeError("frozen learned projection not restored EXACTLY")
        restored_prediction=predict_batch(model,batch)[:,-1]
        torch.testing.assert_close(restored_prediction,inp_original,atol=0,rtol=0)
    return (inp_original,zeroed[:,-1],lesions,
            dict(average_matrix_abs_after_source_events=float(
                     memory[:,-1].abs().mean()),
                 average_original_matrix_abs_all_steps=sum(norms)/len(norms),
                 mean_absolute_stream_prediction_change_if_read_memory_cleared=
                     float((zeroed[:,-1]-inp_original).abs().mean()),
                 saved_restored_stream_forward_exact=True,
                 fitted_batch_versus_stream_parity=True,
                 chronological_count=int(restored.count),
                 clock_mean_hours=float(restored.elapsed_clock.mean()),
                 source_current_age_equals_delta_max_discrepancy=time_dupe))

def run(original_source:Path):
    torch.set_num_threads(1)
    cfg=json.loads(REG.read_text())
    assert_contract(cfg)
    d=read_source(original_source)
    train,_=chronology(d)
    plan=freeze(original_source)
    receipt=json.loads((ORIGINAL_OUT/"TRAIN_RECEIPT.json").read_text())
    check_original_weights(receipt,plan)
    selected,cohort_info=cohort(d,train)
    baseline_per_seed=[]
    for receipt_row in receipt["original_checkpoints"]:
        seed=receipt_row["seed"]
        pth=ORIGINAL_OUT/f"seed-{seed}-full-original.pt"
        if file_hash(pth)!=receipt_row["checkpoint_SHA256"]:
            raise RuntimeError("original trained checkpoint hash changed")
        saved=torch.load(pth,map_location="cpu",weights_only=False)
        model=modelspec(seed)
        model.load_state_dict(saved["model_state"],strict=True)
        model.eval()
        sums={"original":dict(observed_channels=0,observed_pm25=0,
                               absolute_channel_error_sum=0.,absolute_pm25_error_sum=0.)}
        for key in (*ARMS.keys(),"zero_historical_associative_read",
                    "real_last_observation_persistence"):
            sums[key]=dict(observed_channels=0,observed_pm25=0,
                           absolute_channel_error_sum=0.,absolute_pm25_error_sum=0.)
        shifts={key:0. for key in sums if key!="original"}
        state_checks=[]
        samples=0
        for idx in range(0,len(selected),4):
            batch=batch_from_cases(d,selected[idx:idx+4])
            original,no_mem,lesions,checks=source_probe(model,batch)
            state_checks.append(checks)
            variants={
                "original":original,
                "zero_historical_associative_read":no_mem,
                "real_last_observation_persistence":batch[0][:,-1],
                **lesions,
            }
            for name,pred in variants.items():
                observed=metrics(pred,batch)
                for field,value in observed.items():
                    sums[name][field]+=value
                if name!="original":
                    shifts[name]+=float((pred-original).abs().sum())
            samples+=len(batch[0])
        if samples!=64:
            raise RuntimeError("failed to analyze registered 64 authentic source FIT cases")
        compact={}
        for name,vals in sums.items():
            compact[name]=dict(
                FIT_only_normalized_13_channel_MAE=vals["absolute_channel_error_sum"]/
                        vals["observed_channels"],
                FIT_only_normalized_PM25_MAE=vals["absolute_pm25_error_sum"]/
                        vals["observed_pm25"],
                true_observed_physical_channels=vals["observed_channels"],
                true_observed_PM25_targets=vals["observed_pm25"],
                mean_absolute_prediction_shift_from_original_normalized=
                    0.0 if name=="original" else shifts[name]/(samples*14),
            )
        baseline_per_seed.append(dict(
            seed=seed,original_frozen_checkpoint_sha256=receipt_row["checkpoint_SHA256"],
            FIT_source_cases=samples,
            mean_associative_matrix_abs_after_12_observations=sum(
                x["average_matrix_abs_after_source_events"] for x in state_checks
            )/len(state_checks),
            FIT_mean_read_lesion_output_change=sum(
                x["mean_absolute_stream_prediction_change_if_read_memory_cleared"]
                for x in state_checks)/len(state_checks),
            FIT_allwrite_age_equals_dt_max_discrepancy=max(
                x["source_current_age_equals_delta_max_discrepancy"] for x in state_checks),
            source_clock_after_12_event_average_hours=sum(
                x["clock_mean_hours"] for x in state_checks)/len(state_checks),
            true_model_runs_are_12_event_reset_windows=True,
            stored_state_serialization_check="PASSED_exact_value_restoration",
            fitted_batch_vs_stream_check="PASSED_tolerant_numerical_consistency",
            arms=compact,
        ))
        print("PAR_TIME_004_FROZEN_FIT_SEED="+json.dumps(
            baseline_per_seed[-1],sort_keys=True,allow_nan=False),flush=True)
    mean_by_arm={}
    for name in baseline_per_seed[0]["arms"]:
        mean_by_arm[name]={
            "mean_FIT_normalized_13_channel_MAE":sum(
               s["arms"][name]["FIT_only_normalized_13_channel_MAE"]
               for s in baseline_per_seed)/len(baseline_per_seed),
            "mean_abs_output_shift_from_original":sum(
               s["arms"][name]["mean_absolute_prediction_shift_from_original_normalized"]
               for s in baseline_per_seed)/len(baseline_per_seed),
        }
    out={
       "study_id":"PAR-TIME-004-FROZEN-FIT-ONLY-TEMPORAL-AND-MEMORY-LESIONS",
       "scientific_status":"SOURCE_TRAINING_PERIOD_DIAGNOSTIC_NOT_GENERALIZATION",
       "no_training_or_new_checkpoint":True,
       "original_source_SHA256":SHA256,
       "original_checkpoints_SHA_identical":True,
       "authentic_source_training_period_cases":64,
       "source_case_registry":cohort_info,
       "original_model_count":5,
       "trained_on_short_12_event_windows_only":True,
       "temporal_age_confound_all_source_events_write":True,
       "no_independent_validation_or_matched_architecture_superiority":True,
       "original_2014_development_outputs_unread":True,
       "year_2015_2017_numerical_source_outputs_unread":True,
       "per_seed":baseline_per_seed,
       "average_by_arm":mean_by_arm,
    }
    if OUT.exists():
        raise RuntimeError("original attribution artifact was created concurrently; no overwrite")
    OUT.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("PAR_TIME_004_FIT_DIAGNOSTIC_COMPLETE="+json.dumps({
         "arm_mean":mean_by_arm,"original_model_count":5,
         "mean_matrix_norm":sum(s["mean_associative_matrix_abs_after_12_observations"]
                               for s in baseline_per_seed)/5,
         "no_generalization_or_ability_promotion":True,
    },sort_keys=True,allow_nan=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    args=p.parse_args()
    run(args.source)
