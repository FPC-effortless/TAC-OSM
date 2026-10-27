#!/usr/bin/env python3
"""GP-UNC-005: real trained ME111 native neural to GP issue boundary, NO FIT.

Five SHA-checked original ME111 104,663-parameter FP32 source-trained models;
authentic UCI501 original FIT 2013–2014 station inputs only. NO outcomes,
target labels, GP updates, stochastic optimizer, protected 2016-17, or
2015 already-exposed REPORT. Checks native mean and scale untouched by the
truth-free GP issue boundary. This is a source-forward interface test, NOT
GP improvement, independent forecasting qualification, or source-availability
proof. Both original checkpoint and original source are pinned by immutable SHA.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
NATIVE_SHA="77c753a69f10fa13cd7fdf849edb12bc53f03e6d"
SOURCE_SHA="d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8"
SEEDS=(111001,111002,111003,111004,111005)
ORIGINS=(12000,12064)
HISTORY=64
SOURCE_STATIONS=12
FORECAST_CHANNELS=6


def source_checkout(root):
    if not (root/"scripts/native_latent_001_diagnosis.py").is_file():
        raise ValueError("original neural diagnosis source checkout absent")
    head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if head != NATIVE_SHA:
        raise ValueError("source Git HEAD differs from original pinned trained model")
    return head


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pinned-native-checkout",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    if args.out.exists():
        raise ValueError("do not overwrite original integration receipt")
    native=args.pinned_native_checkout.resolve()
    source_checkout(native)
    sys.path[:0]=[str(native),str(native/"src"),str(ROOT/"src")]
    import numpy as np
    import torch
    import tac_osm
    # Extend the original source-owned package strictly with GP diagnostic
    # submodules; the original ME111 native operators resolve from pinned source.
    tac_osm.__path__.append(str(ROOT/"src"/"tac_osm"))
    from scripts import native_latent_001_diagnosis as forensic
    from scripts import me111_conditional_training as original
    from tac_osm.gp_event_boundary_nofit import replay_truth_isolated_constructed

    torch.set_num_threads(1)
    d=original.fit_data()
    if d.get("phase")!="fit" or original.data.SHA != SOURCE_SHA:
        raise ValueError("only publisher original FIT observation source allowed")
    if hashlib.sha256(original.data.ARCHIVE.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError("original UCI501 publisher source hash changed")
    if ORIGINS[-1] + 6 >= original.data.FIT_STOP:
        raise ValueError("original observation times not safely inside FIT")

    checkpoint_hashes=forensic.checked_index(original)
    if set(checkpoint_hashes)!=set(SEEDS):
        raise ValueError("original ME111 checkpoint seed enumeration altered")
    rows=[]
    for seed in SEEDS:
        model,path,actual=forensic.model_for(original,d,seed,checkpoint_hashes[seed])
        original_versions=model.versions()
        original_hash_before=hashlib.sha256(path.read_bytes()).hexdigest()
        all_events=[]
        all_base=[]
        all_scales=[]
        future_only_poison_checked=0
        for t in ORIGINS:
            frame=forensic.read_frame(d,[(station,t) for station in range(SOURCE_STATIONS)],HISTORY)
            with torch.no_grad():
                forecast,state=model(*frame[:6],return_trace=False)
            # Original FIT loader pre-parses FIT, including future source
            # records. Test the actual inference-frame producer rather than
            # falsely claiming data was never loaded into source memory.
            # Only original future t+6 values are poisoned, AFTER the frozen
            # normalization/checkpoint and original frame are computed.
            poisoned=dict(d)
            poisoned["raw"]=d["raw"].copy()
            poisoned["values"]=d["values"].copy()
            poisoned["raw"][:,t+6,0] += 2500.
            poisoned["values"][:,t+6,0] += 5.
            poisoned_frame=forensic.read_frame(
                poisoned,[(station,t) for station in range(SOURCE_STATIONS)],HISTORY)
            if not all(torch.equal(clean,alternate)
                       for clean,alternate in zip(frame[:6],poisoned_frame[:6])):
                raise ValueError("future-only original FIT source values entered current inference inputs")
            with torch.no_grad():
                poisoned_forecast,_=model(*poisoned_frame[:6],return_trace=False)
            if (not torch.equal(forecast.log_location,poisoned_forecast.log_location)
                    or not torch.equal(forecast.left_scale,poisoned_forecast.left_scale)
                    or not torch.equal(forecast.right_scale,poisoned_forecast.right_scale)):
                raise ValueError("future-only source target poisoned original neural forecast")
            future_only_poison_checked+=SOURCE_STATIONS*FORECAST_CHANNELS
            if (forecast.log_location.shape!=(SOURCE_STATIONS,FORECAST_CHANNELS)
                    or forecast.left_scale.shape!=(SOURCE_STATIONS,FORECAST_CHANNELS)
                    or forecast.right_scale.shape!=(SOURCE_STATIONS,FORECAST_CHANNELS)):
                raise ValueError("native trained source changed mean/scale law")
            if (not torch.isfinite(forecast.log_location).all()
                    or not torch.isfinite(forecast.left_scale).all()
                    or not torch.isfinite(forecast.right_scale).all()
                    or (forecast.left_scale<=0).any()
                    or (forecast.right_scale<=0).any()):
                raise ValueError("fitted neural output is nonfinite/nonpositive scale")
            # Frozen original 6-channel normalized-log forecast, not physical
            # arithmetic and NOT a new distribution conversion.
            predicted=forecast.log_location.detach().cpu().tolist()
            lscale=forecast.left_scale.detach().cpu().tolist()
            rscale=forecast.right_scale.detach().cpu().tolist()
            for station in range(SOURCE_STATIONS):
                for channel in range(FORECAST_CHANNELS):
                    v=float(predicted[station][channel])
                    entity=f"station-{station:02d}-channel-{channel}"
                    all_events.append({
                        "kind":"issue",
                        "row_id":f"{seed}:{station}:{channel}:{t}",
                        "entity_id":entity,
                        "features":(v,),
                        "base_prediction":v,
                        "feature_at":float(t),
                        "issued_at":float(t),
                        "checkpoint_sha256":actual})
                    all_base.append(v)
                    all_scales.append((float(lscale[station][channel]),
                                       float(rscale[station][channel])))
        inducing={f"station-{s:02d}-channel-{c}":[[0.0],[1.0]]
                  for s in range(SOURCE_STATIONS) for c in range(FORECAST_CHANNELS)}
        output=replay_truth_isolated_constructed(
            all_events,inducing,checkpoint_sha256=actual,
            adaptive_feedback=False,max_pending=200,
            max_records=200,max_entities=128,noise_variance=0.1)
        if len(output)!=len(all_events):
            raise ValueError("neural output row count changed at GP boundary")
        if any((r.corrected_prediction!=base or r.base_prediction!=base
                or r.eligible_past_updates!=0 or
                not math.isfinite(r.gp_observed_variance))
               for r,base in zip(output,all_base)):
            raise ValueError("no-fit GP boundary silently changed a neural prediction")
        if (model.versions()!=original_versions
                or hashlib.sha256(path.read_bytes()).hexdigest()!=original_hash_before):
            raise ValueError("trained source model changed under read-only inspection")
        rows.append({
            "seed":seed,"checkpoint_sha256":actual,
            "trainable_parameters":sum(p.numel() for p in model.parameters()),
            "native_issuer_rows":len(output),
            "native_source_future_only_poison_checks":future_only_poison_checked,
            "native_source_future_only_poison_changed_forecast":False,
            "native_log_forecasts_exactly_preserved":True,
            "original_left_right_scales_positive_and_untouched":True,
            "native_state_carried_fp32_values":152,
            "gp_updated_on_original_labels":False,
            "original_neural_checkpoint_mutated":False,
            "forecast_log_location_min":min(all_base),
            "forecast_log_location_max":max(all_base),
            "original_left_scale_min":min(x[0] for x in all_scales),
            "original_right_scale_min":min(x[1] for x in all_scales)})
        print("GP_NEURAL_SOURCE_SEED_OK="+str(seed),flush=True)
    result={
        "study":"GP-UNC-005",
        "tier":"REAL_ORIGINAL_FIT_INPUT_ONLY_SOURCE_NEURAL_NO_FIT_DIAGNOSTIC",
        "original_native_git_sha":NATIVE_SHA,
        "original_sensor_archive_sha256":SOURCE_SHA,
        "source":"UCI501 Beijing multisite air quality original FIT 2013-2014",
        "original_native_models":5,
        "original_seed_set":list(SEEDS),
        "total_real_source_neural_issues":sum(r["native_issuer_rows"] for r in rows),
        "original_station_count":SOURCE_STATIONS,
        "original_forecast_channels":FORECAST_CHANNELS,
        "source_fit_hour_origins":list(ORIGINS),
        "original_context_hours":HISTORY,
        "original_fit_archive_preparsed_including_future_fit_values":True,
        "future_numeric_target_extracted_for_bridge_scoring":False,
        "native_original_source_future_only_poison_checks_total":sum(
            row["native_source_future_only_poison_checks"] for row in rows),
        "gp_residual_updates":0,
        "protected_report_and_test_access":False,
        "training_or_optimizer_steps":0,
        "full_original_neural_graph_to_gp_nofit_parity":True,
        "native_uncertainty_distribution_improved":False,
        "independent_test_performed":False,
        "scientific_promotion":"NOT_QUALIFIED",
        "models":rows}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
    print("GP_UNC_005_RESULT="+json.dumps({k:v for k,v in result.items() if k!="models"},sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
