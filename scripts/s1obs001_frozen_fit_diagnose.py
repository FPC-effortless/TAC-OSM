#!/usr/bin/env python3
"""S1OBS001: run PR422 universal observatory on REAL original frozen System1.

Strictly read-only, genuine original FIT Parquet only, SHA-pinned checkpoint
bytes from S1MOTION005 run 40283616062. No source DEV or TEST file opened.
The diagnostician must NEVER claim that local source derivatives/lesions
establish generalization, memory retrieval competency or autonomous safety.
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]

from tac_osm.research_probe import inspect_variant
from tac_osm.research_anticheat import (
    audit_input_boundary, audit_partitions, probe_future_invariance,
    AuditLedger, Finding,
)
from tac_osm.research_torch_basics import gradient_conflict_atlas
from tac_osm.research_system1_adapter import system1_velocity_adapter,SOURCE_FIELDS
from scripts.s1motion005_real_screwdriver_fit import (
    LOCK, FIT, DEV, SEEDS, episode_data, approved_source_metadata, original_models
)
from scripts.s1motion004_real_towel_fit import encode_source_episode
import numpy as np

ORIGINAL_LEDGER=ROOT/"research/parallel-six/S1MOTION005-ORIGINAL-LEDGER.json"
ARMS=("tac_shared","tac_isolated","tac_projected")
COMPONENTS={
    "encoder":"core.core.encoder",
    "affine_memory_gates":"core.core.memory_gates",
    "point_readout":"core.core.readout",
    "observational_auxiliary_head":"core.reactive_head",
}


def loss_gradient_diagnosis(model, x,u,y,event,previous):
    """Physical next-state, proxy BCE and NLL gradients on real FIT."""
    model.eval()
    f=model(x,u,initial_previous_observation=previous)
    point=(f.next_observation-y).square().mean()
    proxy=.1*F.binary_cross_entropy_with_logits(f.adverse_event_logit,event)
    nll=.01*(.5*((f.next_observation-y)/f.residual_scale).square()
               +f.residual_scale.log()).mean()
    names={k:v for k,v in model.named_parameters() if v.requires_grad}
    params=list(names.values())
    losses={"point":point,"motion_proxy":proxy,"uncertainty_nll":nll}
    gs={}
    for label,loss in losses.items():
        grad=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        gs[label]={n:g.detach().double() if g is not None else None
                   for n,g in zip(names,grad)}
    report={}
    for kind,prefix in COMPONENTS.items():
        names0=[n for n in names if n.startswith(prefix)]
        block={}
        for obj in losses:
            block[obj+"_norm"]=math.sqrt(sum(
                float((gs[obj][n]**2).sum()) for n in names0
                if gs[obj][n] is not None))
        for obj in ("motion_proxy","uncertainty_nll"):
            shared=[(gs["point"][n],gs[obj][n]) for n in names0
                    if gs["point"][n] is not None and gs[obj][n] is not None]
            dot=sum(float((a*b).sum()) for a,b in shared)
            a=math.sqrt(sum(float((a*a).sum()) for a,b in shared))
            b=math.sqrt(sum(float((b*b).sum()) for a,b in shared))
            block["point_vs_"+obj+"_cosine"]=(dot/a/b if a>0 and b>0 else None)
            block["point_vs_"+obj+"_dot"]=dot
            block["intersecting_trainable_tensors"]=len(shared)
        report[kind]=block
    global_atlas=gradient_conflict_atlas(model,{
        "state_prediction_point":point,
        "measured_motion_proxy":proxy,
        "predictive_uncertainty_nll":nll})
    return dict(local_block_mechanics=report,
        generic_pr422_gradient_conflict_atlas=global_atlas,
        fit_only_loss_scalars={k:float(v.detach()) for k,v in losses.items()},
        source_label="real_measured_motion_proxy_NOT_actual_robot_harm",
        no_parameter_updates=True)


def real_sample(ep_record, window_length=12):
    return {
        "observation_history":ep_record["x"][:window_length][None],
        "prior_logged_action_history":ep_record["prior_u"][:window_length][None],
        "preceding_measured_observation":ep_record["previous_x"][:1],
    }


def run(original_dir:Path, cache:Path, out:Path, *, seeds):
    ledger=json.loads(ORIGINAL_LEDGER.read_text())
    lock=json.loads(LOCK.read_text())
    approved_source_metadata(lock)
    first=json.loads((original_dir/"BEFORE-DEV-ALL-FROZEN.json").read_text())
    if (not first["all_15_model_optimizer_RNG_state_blobs_SHA_frozen_before_any_DEV_file"]
        or first["source_locked_revision"]!=lock["revision_sha"]
        or first["TEST_ep_ids_never_downloaded"]!=list(range(35,50))):
        raise RuntimeError("original evidence/source governance failed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    # TWO only original FIT episodes. Other FIT normalization from original
    # source-authorized before-DEV receipt, not recomputed from unseen labels.
    training={}
    for ep in (0,1):
        locked=next(x for x in lock["fit_files"] if x["episode"]==ep)
        records,_=episode_data(ep,lock,cache,expected_sha=locked["sha256"])
        training[ep]=encode_source_episode(records,first["normalization_FIT_only"])
    norm=first["normalization_FIT_only"]
    scale=torch.tensor(norm["train_delta_scale_normalized"],dtype=torch.float32)
    first=real_sample(training[0])
    second=real_sample(training[1])
    paired={k:torch.cat((first[k],second[k]),dim=0) for k in first}
    samples={
        "paired_genuine_FIT_episodes_0_and_1":paired,
        "single_genuine_FIT_episode_0":first,
    }
    ledger_audit=AuditLedger("S1OBS001:source-real-frozen-2026-10-26",[
        "source_prior_only","fit_dev_test_episode_identifiers","source_future_noninterference",
        "original_checkpoint_integrity","read_only_probe",
    ])
    boundary=audit_input_boundary(samples["first_genuine_FIT_episode"],
                                  sorted(SOURCE_FIELDS),check="source_prior_only")
    ledger_audit.add(boundary)
    partitions=audit_partitions({"FIT":list(FIT),"DEV":list(DEV),
         "TEST":list(range(35,50))},check="fit_dev_test_episode_identifiers")
    ledger_audit.add(partitions)
    decision=samples["first_genuine_FIT_episode"]
    true_y=training[0]["y"][11:12]
    poisoned={"inputs":decision,"actual_future_outcome":true_y+11}
    normal={"inputs":decision,"actual_future_outcome":true_y}
    no_future=probe_future_invariance(normal,poisoned,
         lambda v:v["inputs"],lambda v:v["actual_future_outcome"],
         check="source_future_noninterference")
    ledger_audit.add(no_future)
    rows=[]
    all_hash_ok=True
    from copy import deepcopy
    for seed in seeds:
        models=original_models(seed,scale)
        for arm in ARMS:
            record=next(x for x in ledger["all_original_model_fit_receipts"]
                        if x["arm"]==arm and x["seed"]==seed)
            filename=original_dir/f"{arm}-{seed}.pt"
            blob=filename.read_bytes()
            digest=hashlib.sha256(blob).hexdigest()
            if digest!=record["model_SHA256"]:
                raise RuntimeError(f"original checkpoint tampered: {seed}/{arm}")
            ck=torch.load(filename,map_location="cpu",weights_only=False)
            if (ck["source_revision"]!=lock["revision_sha"] or
                    ck["seed"]!=seed or ck["arm"]!=arm or
                    ck["updates"]!=192 or ck["source_FIT_checksums"]!=[
                        r["sha256"] for r in lock["fit_files"]]):
                raise RuntimeError("original model source/seed/optimizer receipt invalid")
            model=models[arm]
            model.load_state_dict(ck["model"],strict=True)
            model.eval()
            before={k:v.detach().clone() for k,v in model.state_dict().items()}
            for id_,sample in samples.items():
                probe=inspect_variant(model,sample,system1_velocity_adapter,
                                      isolation=True,profile=True)
                length=sample["observation_history"].shape[1]
                episode_ids=(0,1) if id_.startswith("paired_") else (0,)
                x=sample["observation_history"]
                u=sample["prior_logged_action_history"]
                y=torch.cat((training[e]["y"][:length][None]
                             for e in episode_ids),dim=0)
                event=torch.cat((training[e]["event"][:length][None]
                                 for e in episode_ids),dim=0)
                prev=sample["preceding_measured_observation"]
                gradients=loss_gradient_diagnosis(model,x,u,y,event,prev)
                # Meaningfully separate point forecast from trivially strong
                # deterministic velocity baseline. Look for memory sensitivity
                # INSIDE the learned correction and proxy heads.
                with torch.no_grad():
                    f=model(x,u,initial_previous_observation=prev)
                    velocity=2*x-torch.cat((prev[:,None],x[:,:-1]),dim=1)
                    real_residual=y-velocity
                    correction=f.next_observation-velocity
                    alignment=float((correction.double()*real_residual.double()).mean())
                    energy=float(correction.double().square().mean())
                    true_err=float((f.next_observation.double()-y.double()).square().mean())
                    velocity_err=float(real_residual.double().square().mean())
                if any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()):
                    raise RuntimeError("diagnostics modified checkpoint model weights or buffers")
                iso_status=probe["entity_isolation"]["status"]
                allowed_status="PASS" if len(episode_ids)==2 else "INCONCLUSIVE_ONE_ENTITY"
                if (not probe["memory"]["query_does_not_write_state"]
                    or iso_status!=allowed_status):
                    raise RuntimeError(
                        f"unexpected state purity/entity isolation: {id_}, "
                        f"status={iso_status}, expected={allowed_status}")
                # INCONCLUSIVE_ONE_ENTITY is a coverage limit, NOT a passed
                # isolation check. The B=2 authentic replay MUST pass.
                row=dict(seed=seed,arm=arm,authentic_FIT_episode=id_,
                         original_checkpoint_SHA256=digest,
                         generic_probe=probe,
                         generic_gradient_and_loss_diagnosis=gradients,
                         point_state_relative_residual=dict(
                             learned_correction_energy=energy,
                             signed_alignment_with_actual_next_state_residual=alignment,
                             source_normalized_forecast_MSE=true_err,
                             source_normalized_causal_velocity_MSE=velocity_err),
                         original_3_seed_DEV_gate_unchanged="FAIL_2_OF_3",
                         analysis_fits_or_updates=0)
                rows.append(row)
                memory=probe["memory"]["interventions"]
                m=lambda k: memory[k]["head_mean_absolute_change"]
                print("S1OBS001_ACTUAL_CHECKPOINT_DIAGNOSTIC",json.dumps(
                    dict(seed=seed,arm=arm,source_FIT_episode=id_,
                         entity_isolation_coverage=iso_status,
                      lesion_deltas={
                          k:m(k) for k in ("fast_running_sum",
                         "associative_memory","previous_measured_observation")},
                      memory_grad_to_next=probe["dependencies"]["heads"][
                          "next_measured_state_forecast"]["associative_memory"],
                      memory_grad_to_proxy=probe["dependencies"]["heads"][
                          "motion_proxy_logit"]["associative_memory"],
                      loss_gradient_components=gradients["local_block_mechanics"],
                      source_residual_math=row["point_state_relative_residual"]),
                    sort_keys=True),flush=True)
    ledger_audit.add(Finding("original_checkpoint_integrity","PASS",
        "all selected original 15-study frozen checkpoint source SHAs verified",
        dict(selected_seeds=list(seeds),checked_models=3*len(seeds))))
    ledger_audit.add(Finding("read_only_probe","PASS",
        "no optimizer call, parameter or buffer alteration in local probes",{}))
    report_audit=ledger_audit.report()
    if report_audit["status"]!="PASS":
        raise RuntimeError("required anticheat source-integrity gate failed")
    out.mkdir(parents=True,exist_ok=True)
    full=dict(schema="S1OBS001/real-FIT-original-native-v1",
              source_original_real_episode_FIT_only=[0,1],
              original_15_study_run=40283616062,
              original_15_study_model_artifact=12632762763,
              observational_only_no_action_authority=True,
              source_real_label_as_event_proxy_only=True,
              no_DEV_or_TEST_episode_downloads=True,
              no_optimizer_calls_or_checkpoint_mutation=True,
              model_diag_results=rows,
              source_anticheat_ledger=report_audit,
              full_deployment_qualification=False,
              original_S1MOTION005_scientific_gate="FAIL_2_OF_3")
    (out/"S1OBS001-FROZEN-FIT-DIAGNOSTICS.json").write_text(json.dumps(
        full,indent=2,sort_keys=True)+"\n")
    print("S1OBS001_NATIVE_SYSTEM1_OBSERVATORY_COMPLETE=PASS",flush=True)
    print("S1OBS001_MODEL_SKILL_PROMOTION=NO",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--original-dir",required=True)
    parser.add_argument("--cache",default="/tmp/s1obs001-source-FIT-only")
    parser.add_argument("--out",default="research/diagnostics/s1obs001-frozen-FIT")
    parser.add_argument("--seeds",default="29001,29002,29003")
    a=parser.parse_args()
    wanted=tuple(int(x) for x in a.seeds.split(","))
    if not wanted or any(x not in SEEDS for x in wanted) or len(set(wanted))!=len(wanted):
        raise ValueError("original authorized frozen seeds required")
    run(Path(a.original_dir),Path(a.cache),ROOT/a.out,seeds=wanted)
