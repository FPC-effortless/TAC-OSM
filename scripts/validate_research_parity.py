#!/usr/bin/env python3
"""Fail-closed TAC-OSM architecture comparison parity and provenance gate.

Preflight runs BEFORE fitting, final runs on actual instrumented training
receipts BEFORE reading or claiming heldout architectural-superiority results.
This is a resource-integrity validator, not itself evidence of model skill.
Mandatory matched variables are trainable parameters, identical real data,
matched training/tuning COMPUTE budgets and identical evaluation conditions.
Historical v1 manifests record CPU as efficiency metadata. Prospective v2
manifests enforce the owner's additional <=5% measured CPU parity contract.
It never accepts claimed 'matched' labels in place of numerical receipts.

Usage:
 python scripts/validate_research_parity.py --manifest PATH --phase preflight
 python scripts/validate_research_parity.py --manifest PATH --phase final
 python scripts/validate_research_parity.py --check-changed origin/master...HEAD
"""
from __future__ import annotations
import argparse
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HEX64 = re.compile(r"^[a-fA-F0-9]{64}$")
STUDY_FILE = re.compile(r"(?:^|/)(?:me|dev)(\d{2,3})(?:[-_./]|$)",re.I)
SCHEMA = "tac-osm.parity.v1"
STRICT_SCHEMA = "tac-osm.parity.v2"
PARAM_TOLERANCE=0.01
FLOP_TOLERANCE=0.01
CPU_TOLERANCE=0.05

class ParityError(ValueError): pass

def demand(condition, message):
    if not condition: raise ParityError(message)

def digest(value,name):
    demand(isinstance(value,str) and HEX64.fullmatch(value),name+" must be real 64-char SHA256 evidence")

def positive_num(value,name,whole=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ParityError(name+" missing/not numerical or nonfinite")
    demand(value>0,name+" must be positive")
    if whole:demand(isinstance(value,int),name+" must be integer")

def text_field(value,name):
    demand(isinstance(value,str) and len(value.strip())>=8,name+" missing meaningful value")

def close_enough(entries,key,tolerance,label):
    values=[float(row[key]) for row in entries]
    lo=min(values);hi=max(values)
    demand((hi-lo)/hi<=tolerance,
        label+f" mismatch: min={lo}, max={hi}, max allowable gap={tolerance:.0%}")

def shared(rows,key,seed):
    values=[r[key] for r in rows]
    demand(all(v==values[0] for v in values),
        f"seed {seed}: {key} differs across architecture arms")


def check_original_real_source_registration(m,phase):
    """Owner-authorized same-parameters/data/optimizer-budget certification.

    Real original episode and actual FIT sampled-window digests cannot exist
    until the publisher source is read. Validate the immutable prefit plan
    here; original runner independently verifies source and sampled-order
    hashes BEFORE the first optimizer step and holds all checkpoints before
    opening TEST. Do not manufacture unavailable full FLOP measurements.
    """
    demand(m.get("study_type")=="architecture_comparison","genuine model comparison required")
    demand(m.get("registered_before_first_fit") is True,"preregistration required")
    digest(m.get("dataset_fit_receipt_sha256"),"original source-FIT archive")
    cohorts=m.get("frozen_episode_split_sha256",{})
    for key in ("FIT","DEV","TEST"): digest(cohorts.get(key),key+" cohort")
    seeds=m.get("seeds",[])
    demand(isinstance(seeds,list) and len(seeds)>=2 and
           len(set(seeds))==len(seeds),"multiple distinct original paired seeds required")
    counts=m.get("parameter_counts",{})
    demand(set(counts)=={"native_tac","gru_world"},"two complete competitors required")
    for name,n in counts.items():positive_num(n,name+" active params",whole=True)
    demand((max(counts.values())-min(counts.values()))/max(counts.values())<=PARAM_TOLERANCE,
           "active model parameters differ beyond compulsory 1%")
    budget=m.get("training_budget",{})
    for key in ("updates","batch_size","history_frames","backprop_frames"):
        positive_num(budget.get(key),key,whole=True)
    demand(budget.get("tuning_trials")==0,"identical zero tuning allowance required")
    demand(m.get("same_authentic_training_data") is True and
           m.get("same_real_training_sample_order_within_each_seed") is True,
           "genuine matched original training data/order required")
    demand(m.get("original_test_unopened_until_all_ten_checkpoints_frozen") is True,
           "all real TEST observations must stay sealed until FIT models freeze")
    demand(m.get("cpu_time_matching_required") is False,"CPU speed must not be a parity gate")
    if phase=="final":
        rows=m.get("completed_model_records",[])
        require_total=len(seeds)*len(counts)
        demand(len(rows)==require_total,"missing original trained model checkpoints")
        for seed in seeds:
            paired=[r for r in rows if r.get("seed")==seed]
            demand({r.get("architecture") for r in paired}==set(counts),
                   "missing original matched architecture/seed")
            demand(len({r.get("training_schedule_SHA256") for r in paired})==1,
                   "actual original FIT training windows/order not equal")
            for r in paired:
                digest(r.get("training_schedule_SHA256"),"real source sampling digest")
                digest(r.get("original_checkpoint_sha256"),"trained original checkpoint")
                demand(r.get("active_params")==counts[r["architecture"]],
                       "trained original parameter counts mismatch")
                demand(r.get("original_actual_fit_updates")==budget["updates"],
                       "optimizer training update budget unequal")
                demand(r.get("original_real_history_frames")==budget["history_frames"],
                       "original recurrent temporal exposure unequal")
                demand(r.get("real_backward_steps_per_update")==budget["backprop_frames"],
                       "original BPTT budget unequal")
        demand(m.get("all_frozen_original_models_before_heldout_opening") is True,
               "heldout labels opened before models frozen")
    return {"study_id":m["study_id"],"phase":phase,
            "status":"PASS_OWNER_MANDATORY_PARAMS_DATA_TRAINING_BUDGET",
            "eligible_for_architecture_superiority":
                phase=="final" and m.get("full_fLOP_certification_available") is True,
            "FLOP_exact_consumption_verified":False,
            "CPU_runtime_is_efficiency_only":True}

def check_manifest(m,phase="preflight"):
    demand(isinstance(m,dict),"manifest JSON object required")
    if m.get("source_based_deferred_registration") is True:
        return check_original_real_source_registration(m,phase)
    demand(m.get("schema_version") in (SCHEMA,STRICT_SCHEMA),"unsupported parity schema")
    strict_runtime=m.get("schema_version")==STRICT_SCHEMA
    text_field(m.get("study_id"),"study_id")
    kind=m.get("study_type")
    demand(kind in ("architecture_comparison","noncomparative_diagnostic","capability_development"),
        "study_type must explicitly distinguish architecture comparison from diagnostics")
    if kind=="capability_development":
        # Owner-requested single-architecture construction, not a disguised
        # unmatched winner or a synthetic-only capability experiment.
        demand(m.get("no_architecture_superiority_claim") is True,"development forbids superiority claims")
        demand(m.get("registered_before_first_fit") is True,"development must be registered before fitting")
        text_field(m.get("scientific_purpose"),"development purpose")
        source=m.get("real_world_source",{})
        demand(source.get("observed_real_outcomes") is True,"development requires real outcomes")
        digest(source.get("sha256"),"real source SHA")
        text_field(source.get("license"),"source license")
        demand(m.get("architecture_count")==1,"multiple architectures require comparison parity")
        seeds=m.get("seeds",[])
        demand(len(seeds)>=2 and len(set(seeds))==len(seeds),"multiple distinct development seeds required")
        budget=m.get("training_budget",{})
        for key in ("epochs","batch_size","backprop_events","expected_targets_per_seed"):
            positive_num(budget.get(key),key,whole=True)
        positive_num(m.get("trainable_parameters"),"development parameters",whole=True)
        text_field(m.get("repository_storage_plan"),"original checkpoint storage plan")
        for role in ("FIT","TEMP","CQR","REPORT"):
            digest(m.get("split_sha256",{}).get(role),role+" identity")
        if phase=="final":
            records=m.get("completed_model_records",[])
            demand(len(records)==len(seeds),"missing original development fits")
            demand({r.get("seed") for r in records}==set(seeds),"development seed receipts differ")
            for row in records:
                digest(row.get("checkpoint_sha256"),"original development checkpoint")
                demand(row.get("actual_targets")==budget["expected_targets_per_seed"],"development exposure differs from registration")
                demand(row.get("epochs")==budget["epochs"],"development epochs differ")
                positive_num(row.get("optimizer_cpu_seconds"),"actual training CPU")
        return {"study_id":m["study_id"],"phase":phase,"status":"PASS_SINGLE_ARCHITECTURE_DEVELOPMENT",
                "eligible_for_architecture_superiority":False}
    if kind=="noncomparative_diagnostic":
        demand(m.get("no_architecture_superiority_claim") is True,
            "diagnostic must forbid architecture-superiority claims")
        text_field(m.get("scientific_purpose"),"diagnostic scientific_purpose")
        return {"study_id":m["study_id"],"phase":phase,"eligible_for_architecture_superiority":False,
                "status":"PASS_DIAGNOSTIC_ONLY"}
    demand(m.get("registered_before_first_fit") is True,
        "experiment must have a preregistered comparison before the first fit")
    source=m.get("real_world_source")
    demand(isinstance(source,dict) and source.get("observed_real_outcomes") is True,
        "actual non-synthetic real observations/outcomes required")
    digest(source.get("sha256"),"real_world_source.sha256")
    text_field(source.get("name"),"real_world_source.name")
    text_field(source.get("license"),"real_world_source.license")
    gap=m.get("literature_gap")
    demand(isinstance(gap,dict),"read and document already-established literature before spending compute")
    text_field(gap.get("unresolved_TAConly_question"),"literature_gap.unresolved_TAConly_question")
    text_field(gap.get("what_is_not_being_rediscovered"),"literature_gap.what_is_not_being_rediscovered")
    prior=gap.get("established_work")
    demand(isinstance(prior,list) and len(prior)>=2,"at least two specific prior research works required")
    for p in prior:
        demand(isinstance(p,dict),"literature work must be an object")
        text_field(p.get("paper"),"literature paper")
        text_field(p.get("established_result"),"literature established_result")
        demand(isinstance(p.get("url"),str) and p["url"].startswith("https://"),
            "prior work must have published source URL")
    rows=m.get("arms")
    demand(isinstance(rows,list) and len(rows)>=4,"need >=2 architecture arms x >=2 seeds")
    seed_sets=defaultdict(list)
    for i,r in enumerate(rows):
        demand(isinstance(r,dict),"arm receipt must be object")
        demand(isinstance(r.get("architecture"),str) and len(r["architecture"].strip())>=2,
               f"arms[{i}].architecture requires a nonempty name")
        positive_num(r.get("seed"),f"arms[{i}].seed",whole=True)
        seed_sets[r["seed"]].append(r)
        for key in ("trainable_parameters","deployed_parameters","nontrainable_state_bytes"):
            val=r.get(key)
            if key=="nontrainable_state_bytes":
                demand(isinstance(val,int) and val>=0,"nontrainable_state_bytes must be integer >=0")
            else:positive_num(val,key,whole=True)
        demand(r["deployed_parameters"]>=r["trainable_parameters"],
            "trainable parameters cannot exceed total deployed parameters")
        for k in ("dataset_sha256","train_split_sha256","dev_split_sha256",
                  "test_split_sha256","training_sample_order_sha256",
                  "feature_contract_sha256","target_contract_sha256",
                  "optimizer_schedule_sha256","tuning_policy_sha256",
                  "selection_policy_sha256","evaluation_contract_sha256"):
            digest(r.get(k),f"{r['architecture']} {k}")
        demand(r["dataset_sha256"]==source["sha256"],"arm must use identical original real source")
        text_field(r.get("hardware_profile"),"hardware_profile")
        b=r.get("budget")
        demand(isinstance(b,dict),"each arm must have budget with actual planned compute")
        for k in ("steps","examples_seen","tokens_seen","batch_size","history_events",
                  "planned_training_flops","max_training_flops",
                  "tuning_trials","tuning_flops"):
            if k=="tuning_trials":
                demand(isinstance(b.get(k),int) and b[k]>=0,k+" must be a nonnegative integer")
            elif k=="tuning_flops":
                demand(isinstance(b.get(k),(int,float)) and not isinstance(b.get(k),bool)
                       and math.isfinite(b[k]) and b[k]>=0,k+" must be a nonnegative compute estimate")
            else:positive_num(b.get(k),k,whole=k in ("steps","examples_seen","tokens_seen",
                                                        "batch_size","history_events"))
        text_field(b.get("prefit_profile_method"),"original prefit profiler method")
        if strict_runtime:
            positive_num(b.get("max_optimizer_cpu_seconds"),"max_optimizer_cpu_seconds")
            positive_num(b.get("prefit_profile_optimizer_cpu_seconds"),
                         "prefit_profile_optimizer_cpu_seconds")
        demand(b["planned_training_flops"]<=b["max_training_flops"],
            "planned compute must not exceed matched upper budget")
        if phase=="final":
            result=r.get("measured")
            demand(isinstance(result,dict),"postfit actual measured resource receipt missing")
            for k in ("actual_steps","actual_examples","actual_tokens","training_flops",
                      "optimizer_cpu_seconds","peak_memory_bytes"):
                positive_num(result.get(k),k,
                    whole=k in ("actual_steps","actual_examples","actual_tokens","peak_memory_bytes"))
            text_field(result.get("flops_profiler"),"independent training_flops profiler provenance")
            digest(result.get("checkpoint_sha256"),"frozen checkpoint")
            digest(result.get("actual_sample_order_sha256"),"actual data replay digest")
            digest(result.get("evaluation_split_sha256"),"actual evaluated test IDs")
            demand(result["actual_steps"]==b["steps"] and result["actual_examples"]==b["examples_seen"]
                   and result["actual_tokens"]==b["tokens_seen"],
                   "actual steps/examples/tokens differ from matching preregistered budget")
            demand(result["actual_sample_order_sha256"]==r["training_sample_order_sha256"],
                   "actual training sample order deviated from frozen plan")
            demand(result["evaluation_split_sha256"]==r["test_split_sha256"],
                   "evaluated test IDs differ from frozen plan")
            demand(result["training_flops"]<=b["max_training_flops"],
                   "actual billed FLOPs exceeded the common preregistered ceiling")
            if strict_runtime:
                demand(result["optimizer_cpu_seconds"]<=b["max_optimizer_cpu_seconds"],
                       "actual optimizer CPU exceeded the common preregistered ceiling")
    demand(len(seed_sets)>=2,"need >=2 controlled independent seeds")
    names={r["architecture"] for r in rows}
    demand(len(names)>=2,"need at least two different architectures")
    for seed,group in seed_sets.items():
        demand({r["architecture"] for r in group}==names and len(group)==len(names),
            f"seed {seed} architecture arm matrix missing/duplicated")
        for key in ("dataset_sha256","train_split_sha256","dev_split_sha256",
                    "test_split_sha256","training_sample_order_sha256",
                    "feature_contract_sha256","target_contract_sha256",
                    "optimizer_schedule_sha256","tuning_policy_sha256",
                    "selection_policy_sha256","evaluation_contract_sha256",
                    "hardware_profile"):
            shared(group,key,seed)
        for key in ("steps","examples_seen","tokens_seen","batch_size",
                    "history_events","max_training_flops",
                    "tuning_trials","tuning_flops"):
            shared([r["budget"] for r in group],key,seed)
        if strict_runtime:
            shared([r["budget"] for r in group],"max_optimizer_cpu_seconds",seed)
        close_enough(group,"trainable_parameters",PARAM_TOLERANCE,f"seed {seed} active parameters")
        close_enough(group,"deployed_parameters",PARAM_TOLERANCE,f"seed {seed} deployed parameters")
        shared([r["budget"] for r in group],"prefit_profile_method",seed)
        close_enough([r["budget"] for r in group],"prefit_profile_training_flops",FLOP_TOLERANCE,
                     f"seed {seed} PROFILER prefit FLOPs")
        if strict_runtime:
            close_enough([r["budget"] for r in group],
                         "prefit_profile_optimizer_cpu_seconds",CPU_TOLERANCE,
                         f"seed {seed} PROFILER prefit CPU seconds")
        close_enough([r["budget"] for r in group],"planned_training_flops",FLOP_TOLERANCE,
                     f"seed {seed} planned FLOPs")
        if phase=="final":
            actual=[r["measured"] for r in group]
            close_enough(actual,"training_flops",FLOP_TOLERANCE,f"seed {seed} actual FLOPs")
            if strict_runtime:
                close_enough(actual,"optimizer_cpu_seconds",CPU_TOLERANCE,
                             f"seed {seed} actual CPU seconds")
    # v1 preserves historical interpretation. New ME055 work uses v2, which
    # implements the owner's explicit measured CPU <=5% requirement.
    return {"study_id":m["study_id"],"phase":phase,"matched_seeds":sorted(seed_sets),
            "matched_architectures":sorted(names),"matched_source":source["sha256"],
            "status":"PASS","eligible_for_architecture_superiority":phase=="final"}


def find_changed_studies(base_head):
    command=["git","diff","--name-only",base_head]
    check=subprocess.run(command,check=True,capture_output=True,text=True)
    modified=check.stdout.splitlines();studies=set()
    for path in modified:
        if path.startswith(("src/tac_osm/","scripts/",".github/workflows/",
                            "research/mathematical-engineering/")):
            m=STUDY_FILE.search(path)
            if m:
                studies.add(("me" if "/me" in "/"+path.lower() else "dev")+m.group(1))
    return sorted(studies)


def check_changed(base_head,root):
    studies=find_changed_studies(base_head)
    for study in studies:
        file=root/"research"/"mathematical-engineering"/study/"FAIR_COMPARISON.json"
        demand(file.exists(),
            f"{study}: missing preregistered FAIR_COMPARISON.json; no new experiment permitted")
        manifest=json.loads(file.read_text())
        result=check_manifest(manifest,"preflight")
        if manifest["study_type"] in ("architecture_comparison","capability_development"):
            workflows=list((root/".github/workflows").glob(study+"*.yml"))
            workflows+=list((root/".github/workflows").glob(study+"*.yaml"))
            demand(bool(workflows),
                f"{study}: training workflow missing (preflight must run before fitting)")
            combined="\n".join(p.read_text() for p in workflows)
            demand("validate_research_parity.py --manifest" in combined
                   and "--phase preflight" in combined and "--phase final" in combined,
                   f"{study}: workflow must gate before first fit AND after actual compute measurement")
        print(json.dumps({"study":study,"preflight":result},sort_keys=True))
    return studies


def main():
    cli=argparse.ArgumentParser()
    group=cli.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest")
    group.add_argument("--check-changed",metavar="BASE...HEAD")
    cli.add_argument("--phase",choices=("preflight","final"),default="preflight")
    args=cli.parse_args()
    try:
        if args.manifest:
            r=check_manifest(json.loads(Path(args.manifest).read_text()),args.phase)
            print(json.dumps(r,sort_keys=True))
        else:
            studies=check_changed(args.check_changed,Path.cwd())
            print(json.dumps({"status":"PASS","registered_changed_studies":studies},
                             sort_keys=True))
    except (ParityError,ValueError,KeyError,json.JSONDecodeError,subprocess.CalledProcessError) as e:
        print("RESEARCH_PARITY_HARD_FAILURE: "+str(e),file=sys.stderr)
        return 2
    return 0

if __name__=="__main__":
    sys.exit(main())
