"""PAR-MEM-012: frozen real-source associative READ attribution, NO FIT.

Reconstruct source-verified PAR-MEM-011 original DEV prediction arrays
from exactly SHA-pinned full checkpoint weights, then intervene ONLY at
the FINAL 16D associative read. All legitimate source writes, fast
temporal state, source owner/clock, real observed inputs, decoder weights,
and target values stay fixed. Same frozen checkpoint is evaluated with
correct old source index, adjacent genuinely observed wrong event, far
genuinely observed wrong event and 16 zero memory-read features.

A read-zero intervention is NOT a retrained no-memory baseline, and
none of these source results is independent virgin TEST.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.parmem011_genuine_source_train import (
    ROOT, PROTO as PARENT_PROTO, PLAN as PARENT_PLAN, OUT as PARENT_OUT,
    fixed as original_source,
    frozen_models, neural, source_batch, sha_file,
    SEEDS, ARMS, LAGS, DEV_ANCHORS, T,
)
from scripts.partime002_real_source import SHA256

STUDY="PAR-MEM-012-FROZEN-REAL-HISTORICAL-READ-ATTRIBUTION"
PROTOCOL=ROOT/"research/parallel-six/PAR-MEM-012-PROTOCOL.json"
RESULTS=ROOT/"research/parallel-six/PAR-MEM-012-RESULTS.json"
REPLAY=ROOT/"research/parallel-six/PAR-MEM-012-ORIGINAL-PREDICTIONS.pt"
MODES=("correct_historic_index","adjacent_later_historic_index",
       "distant_later_historic_index","zero_associative_read")

def preregister(*,replay):
    p=json.loads(PROTOCOL.read_text())
    expect=dict(study_id=STUDY,source_zip_SHA256=SHA256,
                seed_ids=list(SEEDS),original_trained_checkpoint_arms=list(ARMS),
                inference_interventions=list(MODES),target_lags=list(LAGS),
                real_DEV_anchors=list(DEV_ANCHORS),real_source_sites=12,
                real_queries_per_lag_per_checkpoint=120,
                training_or_optimizer_steps=0,
                source_model_weights_not_changed=True)
    for k,v in expect.items():
        if p.get(k)!=v:raise RuntimeError("real frozen intervention protocol changed: "+k)
    if RESULTS.exists()!=replay or REPLAY.exists()!=replay:
        raise RuntimeError("original source frozen intervention output must be created once, replay later")

def probe_one_batch(model,batch,lag):
    """Forward only once; derive all alternate decoder inputs from SAME
    frozen pre-current-write state with source-verified known past clocks."""
    with torch.no_grad():
        output,trace=model(
            batch["obs"],batch["action"],
            elapsed_dt=batch["dt"],
            write_mask=torch.ones_like(batch["dt"],dtype=torch.bool),
            source_ids=batch["owners"],event_index=batch["indexes"],
            query_event_index=batch["query"],return_trace=True)
        memory=trace["aligned_before_write"][:,-1]
        last=T-1
        correct_ordinal=batch["indexes"][:,last-lag]
        adjacent_ordinal=batch["indexes"][:,last-lag+1]
        distant_ordinal=batch["indexes"][:,last-lag+max(2,lag//2)]
        if not ((correct_ordinal<adjacent_ordinal).all().item() and
                (adjacent_ordinal<distant_ordinal).all().item() and
                (distant_ordinal<batch["indexes"][:,-1]).all().item()):
            raise RuntimeError("intervention may not access a valid source-observed prior event")
        reads={}
        keysets={}
        for mode,ordinal in (
            (MODES[0],correct_ordinal),
            (MODES[1],adjacent_ordinal),
            (MODES[2],distant_ordinal)):
            reads[mode],_=model._read(memory,ordinal)
            keysets[mode],_=model._keys(ordinal,memory.dtype)
        reads[MODES[3]]=torch.zeros_like(reads[MODES[0]])
        if memory.shape!=(batch["obs"].shape[0],2,16,8):
            raise RuntimeError("unexpected model memory geometry")
        model_read=trace["aligned_old_read"][:,-1]
        torch.testing.assert_close(model_read,
            reads[MODES[0]],rtol=2e-5,atol=2e-6)
        # Forecast comparison is conditional on same frozen trained model,
        # same read-before-write state and original physical input.
        base=model.backbone(
            batch["obs"],batch["action"],elapsed_dt=batch["dt"],
            write_mask=torch.ones_like(batch["dt"],dtype=torch.bool))
        forecast={mode:model._predictive_correction(base[:,-1],read)
                  for mode,read in reads.items()}
        torch.testing.assert_close(forecast[MODES[0]],output[:,-1],
                                   rtol=3e-5,atol=3e-6)
        decoded={mode:model.historical_content(read)[:,0]
                 for mode,read in reads.items()}
        for mode in MODES:
            if not torch.isfinite(decoded[mode]).all().item():
                raise RuntimeError("non-finite frozen historical read")
        norm=torch.linalg.vector_norm(reads[MODES[0]],dim=-1)
        delta_adj=torch.linalg.vector_norm(
            reads[MODES[0]]-reads[MODES[1]],dim=-1)
        delta_far=torch.linalg.vector_norm(
            reads[MODES[0]]-reads[MODES[2]],dim=-1)
        cosine_adj=torch.nn.functional.cosine_similarity(
            keysets[MODES[0]],keysets[MODES[1]],dim=-1).mean(-1)
        return dict(decoded=decoded,forecast=forecast,
                    target=batch["past"].clone(),
                    source_current=batch["obs"][:,-1,0].clone(),
                    future_target=batch["next"][:,0].clone(),
                    future_valid=batch["next_valid"][:,0].clone(),
                    key_correct_adjacent_cos=cosine_adj,
                    correct_read_l2=norm,correct_minus_adjacent_read_l2=delta_adj,
                    correct_minus_distant_read_l2=delta_far)

def run(source:Path,*,replay):
    torch.set_num_threads(1)
    preregister(replay=replay)
    plan,data,_,dev,original=frozen_original=original_source(source)
    trained=frozen_models(plan,original)
    all_arr={}
    records=[]
    for seed in SEEDS:
        for arm in ARMS:
            checkpoint=PARENT_OUT/f"seed-{seed}-{arm}-original.pt"
            receipt=next((r for r in trained["records"]
                         if r["seed"]==seed and r["arm"]==arm),None)
            if receipt is None or sha_file(checkpoint)!=receipt["original_checkpoint_SHA256"]:
                raise RuntimeError("checkpoint bytes modified during frozen evaluation")
            frozen=torch.load(checkpoint,map_location="cpu",weights_only=False)
            model=neural(seed,original)
            model.load_state_dict(frozen["model_state"],strict=True)
            model.eval()
            if sum(p.numel() for p in model.parameters())!=98326:
                raise RuntimeError("original trained 100k model changed")
            arrays={mode:[] for mode in MODES}
            forecasts={mode:[] for mode in MODES}
            others={k:[] for k in (
                "target","source_current","future_target","future_valid",
                "key_correct_adjacent_cos","correct_read_l2",
                "correct_minus_adjacent_read_l2",
                "correct_minus_distant_read_l2")}
            lags=[];identities=[]
            for anchor in DEV_ANCHORS:
                for lag in LAGS:
                    sites=list(range(12))
                    b=source_batch(data,dev,sites,[anchor]*12,lag,ARMS[0])
                    x=probe_one_batch(model,b,lag)
                    for mode in MODES:
                        arrays[mode].append(x["decoded"][mode])
                        forecasts[mode].append(x["forecast"][mode][:,0])
                    for key in others:others[key].append(x[key])
                    lags.extend([lag]*12)
                    identities.extend((site,anchor,lag) for site in sites)
            bundle={
                "seed":seed,"trained_arm":arm,
                "checkpoint_SHA256":receipt["original_checkpoint_SHA256"],
                "source_owner_anchor_lag":torch.tensor(identities,dtype=torch.int64),
                "lag":torch.tensor(lags,dtype=torch.int64),
                **{"decoded_"+mode:torch.cat(arrays[mode]).detach()
                   for mode in MODES},
                **{"forecast_"+mode:torch.cat(forecasts[mode]).detach()
                   for mode in MODES},
                **{key:torch.cat(v).detach() for key,v in others.items()}}
            # Original parent predictions provide *independent historical
            # numerical reference* established before this study was coded.
            prev=torch.load(
                PARENT_OUT/f"seed-{seed}-{arm}-original_DEV_tensors.pt",
                map_location="cpu",weights_only=True)
            torch.testing.assert_close(
                bundle["source_owner_anchor_lag"],prev["source_identity"],
                rtol=0,atol=0)
            torch.testing.assert_close(bundle["target"],prev["truth"],
                                       rtol=0,atol=0)
            torch.testing.assert_close(bundle["source_current"],
                                       prev["source_current"],rtol=0,atol=0)
            native_mode=MODES[0] if arm==ARMS[0] else MODES[1]
            torch.testing.assert_close(
                bundle["decoded_"+native_mode],prev["prediction"],
                rtol=3e-5,atol=3e-6)
            torch.testing.assert_close(
                bundle["forecast_"+native_mode],prev["forecast_PM25"],
                rtol=3e-5,atol=3e-6)
            all_arr[f"{seed}-{arm}"]=bundle
            lag_report={}
            for lag in LAGS:
                mask=bundle["lag"]==lag
                true=bundle["target"][mask]
                if int(mask.sum())!=120:raise RuntimeError("source query-count drift")
                baseline=float((bundle["source_current"][mask]-true).abs().mean())
                metrics={}
                for mode in MODES:
                    pred=bundle["decoded_"+mode][mask]
                    physical_valid=mask & bundle["future_valid"]
                    metrics[mode]=dict(
                        source_historical_normalized_PM25_MAE=float(
                            (pred-true).abs().mean()),
                        paired_abs_prediction_shift_vs_zero_read=float(
                            (pred-bundle["decoded_"+MODES[3]][mask]).abs().mean()),
                        source_future_PM25_normalized_MAE=float((
                            bundle["forecast_"+mode][physical_valid]-
                            bundle["future_target"][physical_valid]).abs().mean()),
                        paired_abs_forecast_shift_vs_zero_read=float((
                            bundle["forecast_"+mode][physical_valid]-
                            bundle["forecast_"+MODES[3]][physical_valid]).abs().mean()))
                lag_report[str(lag)]=dict(
                    genuine_observed_source_queries=120,
                    source_copy_current_value_normalized_MAE=baseline,
                    original_frozen_interventions=metrics,
                    correct_adjacent_writer_key_cosine=float(
                        bundle["key_correct_adjacent_cos"][mask].mean()),
                    trained_correct_memory_read_l2=float(
                        bundle["correct_read_l2"][mask].mean()),
                    true_wrong_memory_read_l2_difference=float(
                        bundle["correct_minus_adjacent_read_l2"][mask].mean()),
                    true_distant_memory_read_l2_difference=float(
                        bundle["correct_minus_distant_read_l2"][mask].mean()))
            record=dict(seed=seed,original_trained_checkpoint_arm=arm,
                        frozen_original_checkpoint_SHA256=receipt["original_checkpoint_SHA256"],
                        lags=lag_report)
            records.append(record)
            print("PAR_MEM_012_ORIGINAL_REAL_READ_LESION="+json.dumps(
                {"seed":seed,"arm":arm,"lag_64":lag_report["64"]},
                sort_keys=True),flush=True)
    means={}
    for arm in ARMS:
        subset=[x for x in records if x["original_trained_checkpoint_arm"]==arm]
        means[arm]={}
        for lag in LAGS:
            means[arm][str(lag)]={}
            for mode in MODES:
                measure="source_historical_normalized_PM25_MAE"
                scores=[r["lags"][str(lag)]["original_frozen_interventions"][mode][measure]
                        for r in subset]
                zero=[r["lags"][str(lag)]["original_frozen_interventions"][MODES[3]][measure]
                      for r in subset]
                means[arm][str(lag)][mode]=dict(
                    mean_normalized_PM25_MAE=sum(scores)/len(scores),
                    frozen_seed_wins_over_zero=sum(x<y for x,y in zip(scores,zero)),
                    seed_count=len(scores))
    result=dict(
        study_id=STUDY,original_publisher_source_SHA256=SHA256,
        genuine_source_previously_exposed_DEV=True,
        train_updates_in_current_experiment=0,
        original_source_trained_full_models=10,
        original_pretrained_model_and_optimizer_checkpoints_unchanged=True,
        frozen_same_checkpoint_correct_vs_wrong_vs_zero_vs_far_inference_only=True,
        no_trained_stateless_no_read_comparator=True,
        no_qualified_causal_long_term_memory_benefit=True,
        original_source_genuine_record_rows=records,
        original_five_seed_average_per_trained_arm=means)
    if replay:
        old=json.loads(RESULTS.read_text())
        def close(a,b,path="root"):
            if type(a)!=type(b):raise RuntimeError("source attribution replay schema drift "+path)
            if isinstance(a,dict):
                if set(a)!=set(b):raise RuntimeError("source attribution keys drift "+path)
                for k in a:close(a[k],b[k],path+"."+str(k))
            elif isinstance(a,list):
                if len(a)!=len(b):raise RuntimeError("source original count drift "+path)
                for i,(x,y) in enumerate(zip(a,b)):close(x,y,path+f".{i}")
            elif isinstance(a,float):
                if abs(a-b)>1e-5*max(1,abs(a)):
                    raise RuntimeError("source attribution metric drift "+path)
            elif a!=b:raise RuntimeError("source attribution identity drift "+path)
        close(old,result)
        saved=torch.load(REPLAY,map_location="cpu",weights_only=True)
        if set(saved)!=set(all_arr):raise RuntimeError("replay checkpoint keys drift")
        for k in saved:
            if set(saved[k])!=set(all_arr[k]):raise RuntimeError("replay tensor names drift")
            for field,a in all_arr[k].items():
                b=saved[k][field]
                if torch.is_tensor(a):
                    torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-6)
                elif a!=b:raise RuntimeError("original source seeded checkpoint drift")
        print("PAR_MEM_012_COLD_SOURCE_ORIGINAL_FROZEN_ALL_MODEL_REPLAY_PASS",flush=True)
    else:
        RESULTS.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
        torch.save(all_arr,REPLAY)
    print("PAR_MEM_012_SOURCE_READ_INTERVENTION_FIVE_SEED_MEANS="+json.dumps(
        means,sort_keys=True,allow_nan=False),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True,type=Path)
    parser.add_argument("--mode",choices=("audit","replay"),required=True)
    a=parser.parse_args()
    run(a.source,replay=a.mode=="replay")
