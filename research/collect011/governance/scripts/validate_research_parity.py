#!/usr/bin/env python3
"""Fail-closed TAC-OSM architecture comparison parity and provenance gate.

Preflight runs BEFORE fitting, final runs on actual instrumented training
receipts BEFORE reading or claiming heldout architectural-superiority results.
This is a resource-integrity validator, not itself evidence of model skill.
Mandatory matched variables are trainable parameters, identical real data,
matched training/tuning COMPUTE budgets and identical evaluation conditions.
CPU wall-clock/process runtime is recorded for efficiency only; differences
in how fast distinct architectures execute are NOT a fairness-gate failure.
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

# Required for universal signed-evidence gate on all prospective model fits.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HEX64 = re.compile(r"^[a-fA-F0-9]{64}$")
STUDY_FILE = re.compile(r"(?:^|/)(?:me|dev)(\d{2,3})(?:[-_./]|$)",re.I)
SCHEMA = "tac-osm.parity.v1"
PARAM_TOLERANCE=0.01
FLOP_TOLERANCE=0.01
# Prospective model-size policy: all trained neural models ~100k trainables.
# Historical studies remain immutable, not silently retroactively disqualified.
MODEL_TARGET_PARAMETERS = 100_000
MODEL_TARGET_RELATIVE_TOLERANCE = 0.05
MODEL_MIN_PARAMETERS = 95_000
MODEL_MAX_PARAMETERS = 105_000
LEGACY_MAX_ME_NUMBER = 72
LEGACY_MAX_DEV_NUMBER = 61

def requires_100k_policy(study_id):
    """Prospective trained studies; keep already executed historical evidence frozen.

    ME001..ME072 and DEV001..DEV061 were registered before the new constraint.
    Any unrecognized/new study ID follows the new policy, not a bypass.
    """
    if not isinstance(study_id, str):
        return True
    m = re.match(r"^\s*(ME|DEV)[-_ ]?(\d{1,3})(?:\b|[_-])", study_id, re.I)
    if m:
        cutoff = LEGACY_MAX_ME_NUMBER if m.group(1).upper() == "ME" else LEGACY_MAX_DEV_NUMBER
        return int(m.group(2)) > cutoff
    return True

def check_model_parameter_target(trainable_parameters, deployed_parameters=None, *, label="model"):
    """Hard prefit and postfit count guard for ALL new trained neural models.

    Count trained encoder/memory/decoder/router/uncertainty/action heads. Include
    frozen learned model weights in deployed counts; exclude nonlearned buffers.
    """
    positive_num(trainable_parameters, f"{label}.trainable_parameters", whole=True)
    demand(MODEL_MIN_PARAMETERS <= trainable_parameters <= MODEL_MAX_PARAMETERS,
           f"{label}: prospective 100k target requires 95,000..105,000 trainable parameters, got {trainable_parameters}")
    if deployed_parameters is not None:
        positive_num(deployed_parameters, f"{label}.deployed_parameters", whole=True)
        demand(deployed_parameters >= trainable_parameters,
               f"{label}: deployed learned parameters cannot be below trainable count")
        demand(MODEL_MIN_PARAMETERS <= deployed_parameters <= MODEL_MAX_PARAMETERS,
               f"{label}: 100k target requires 95,000..105,000 deployed learned parameters, got {deployed_parameters}")


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

def _check_manifest_parity_only(m,phase="preflight"):
    demand(isinstance(m,dict),"manifest JSON object required")
    demand(m.get("schema_version")==SCHEMA,"unsupported parity schema")
    text_field(m.get("study_id"),"study_id")
    kind=m.get("study_type")
    demand(kind in ("architecture_comparison","single_model_training","noncomparative_diagnostic"),
        "study_type must distinguish comparison, single-model fit and no-fit diagnostics")
    if kind=="noncomparative_diagnostic":
        demand(m.get("no_architecture_superiority_claim") is True,
            "diagnostic must forbid architecture-superiority claims")
        text_field(m.get("scientific_purpose"),"diagnostic scientific_purpose")
        if requires_100k_policy(m["study_id"]):
            demand(m.get("no_new_model_fits") is True,
                   "prospective noncomparative diagnostics MUST be explicitly no-fit; use single_model_training for a new model")
        return {"study_id":m["study_id"],"phase":phase,"eligible_for_architecture_superiority":False,
                "status":"PASS_DIAGNOSTIC_ONLY"}
    if kind=="single_model_training":
        demand(m.get("registered_before_first_fit") is True,
               "single-model fit must be preregistered")
        source=m.get("real_world_source")
        demand(isinstance(source,dict) and source.get("observed_real_outcomes") is True,
               "authentic real source outcomes required")
        digest(source.get("sha256"),"real_world_source.sha256")
        models=m.get("models")
        demand(isinstance(models,list) and len(models)>=1,
               "single-model training requires >=1 declared model/seed")
        for i,model in enumerate(models):
            demand(isinstance(model,dict),f"models[{i}] must be an object")
            if requires_100k_policy(m["study_id"]):
                check_model_parameter_target(model.get("trainable_parameters"),
                                             model.get("deployed_parameters"),
                                             label=f"models[{i}]")
            positive_num(model.get("trainable_parameters"),f"models[{i}].trainable_parameters",whole=True)
            positive_num(model.get("deployed_parameters"),f"models[{i}].deployed_parameters",whole=True)
            positive_num(model.get("seed"),f"models[{i}].seed",whole=True)
            if phase=="final":
                receipt=model.get("measured")
                demand(isinstance(receipt,dict),
                       f"models[{i}] final actual measurement receipt required")
                if requires_100k_policy(m["study_id"]):
                    check_model_parameter_target(receipt.get("trainable_parameters"),
                                                 receipt.get("deployed_parameters"),
                                                 label=f"models[{i}].measured")
                demand(receipt.get("trainable_parameters")==model["trainable_parameters"],
                       f"models[{i}] actual trainable count differs from frozen preregistration")
                digest(receipt.get("checkpoint_sha256"),f"models[{i}] final checkpoint")
        return {"study_id":m["study_id"],"phase":phase,
                "eligible_for_architecture_superiority":False,
                "status":"PASS_SINGLE_MODEL" if phase=="final" else "PASS_PREFIT_SINGLE_MODEL"}
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
        if requires_100k_policy(m["study_id"]):
            check_model_parameter_target(r["trainable_parameters"], r["deployed_parameters"],
                                         label=f"arms[{i}] {r['architecture']}")
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
            if requires_100k_policy(m["study_id"]):
                check_model_parameter_target(
                    result.get("trainable_parameters"),result.get("deployed_parameters"),
                    label=f"postfit {r['architecture']} seed {r['seed']}")
                demand(result["trainable_parameters"]==r["trainable_parameters"] and
                       result["deployed_parameters"]==r["deployed_parameters"],
                       "postfit actual model parameter counts deviate from registered original prefit counts")
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
        close_enough(group,"trainable_parameters",PARAM_TOLERANCE,f"seed {seed} active parameters")
        close_enough(group,"deployed_parameters",PARAM_TOLERANCE,f"seed {seed} deployed parameters")
        shared([r["budget"] for r in group],"prefit_profile_method",seed)
        close_enough([r["budget"] for r in group],"prefit_profile_training_flops",FLOP_TOLERANCE,
                     f"seed {seed} PROFILER prefit FLOPs")
        close_enough([r["budget"] for r in group],"planned_training_flops",FLOP_TOLERANCE,
                     f"seed {seed} planned FLOPs")
        if phase=="final":
            actual=[r["measured"] for r in group]
            close_enough(actual,"training_flops",FLOP_TOLERANCE,f"seed {seed} actual FLOPs")
    # Absolute 100k ±5% arm checks apply independently of the matched-arm 1% gate.
    # No CPU-time parity test: real execution speed is an efficiency result,
    # not a requirement for equal data, parameters or compute allotments.
    return {"study_id":m["study_id"],"phase":phase,"matched_seeds":sorted(seed_sets),
            "matched_architectures":sorted(names),"matched_source":source["sha256"],
            "status":"PASS","eligible_for_architecture_superiority":phase=="final"}



def check_manifest(m, phase="preflight", *, reward_attestation=None,
                   trusted_reward_public_key_hex=None):
    """Canonical prospective gate: resource parity PLUS independently signed
    per-variant reward/outcome/access evidence. Historical studies unchanged.

    No argument can turn enforcement off. Test-only constructed receipts are
    not genuine source evidence. For CLI runs the public key comes exclusively
    from the operator-provisioned environment, never the model manifest.
    """
    report = _check_manifest_parity_only(m, phase)
    if m["study_type"] == "noncomparative_diagnostic" or not requires_100k_policy(m["study_id"]):
        report["reward_integrity"] = "NOT_APPLICABLE_HISTORICAL_OR_NO_FIT"
        return report
    from tac_osm.research_reward_runner import (
        RewardEvidenceError, verify_reward_attestation)
    try:
        receipt = verify_reward_attestation(
            m, phase, reward_attestation,
            trusted_public_key_hex=trusted_reward_public_key_hex)
    except RewardEvidenceError as exc:
        raise ParityError("independent reward/outcome/access runner gate: " + str(exc)) from exc
    report["reward_integrity"] = receipt
    return report


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
        # Registration records must not claim that an unsigned comparison
        # qualifies for training: the trusted evaluator's preflight receipt is
        # supplied to the runner at execution, not embedded in source files.
        result=_check_manifest_parity_only(manifest,"preflight")
        if manifest["study_type"] in ("architecture_comparison", "single_model_training"):
            demand(requires_100k_policy(manifest["study_id"]) is True,
                   "historical experiment registration is immutable")
        if manifest["study_type"] in ("architecture_comparison","single_model_training"):
            workflows=list((root/".github/workflows").glob(study+"*.yml"))
            workflows+=list((root/".github/workflows").glob(study+"*.yaml"))
            demand(bool(workflows),
                f"{study}: training workflow missing (preflight must run before fitting)")
            combined="\n".join(p.read_text() for p in workflows)
            independent_stage_hooks = (
                "validate_research_parity.py --manifest" in combined
                and "--phase preflight" in combined and "--phase final" in combined
                and combined.count("--reward-evidence") >= 2)
            audited_launcher = (
                "scripts/run_qualified_research.py" in combined
                and "--prefit-evidence" in combined and "--final-evidence" in combined
                and "--prefit-manifest" in combined and "--final-manifest" in combined)
            demand((independent_stage_hooks or audited_launcher)
                   and "TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX" in combined,
                   f"{study}: runner must block optimizer on signed independent "
                   "source/reward/ACL prefit evidence and block postfit promotion "
                   "without signed factual goal/reward/checkpoint evidence")
        print(json.dumps({"study":study,"preflight":result},sort_keys=True))
    return studies


def main():
    cli=argparse.ArgumentParser()
    group=cli.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest")
    group.add_argument("--check-changed",metavar="BASE...HEAD")
    group.add_argument("--model-parameters",type=int,
                       help="Standalone prefit guard on exact counted trainable parameters, even outside matched comparisons")
    cli.add_argument("--deployed-parameters",type=int,
                     help="Optional complete learned parameter count for standalone prefit/final guard")
    cli.add_argument("--phase",choices=("preflight","final"),default="preflight")
    cli.add_argument("--reward-evidence", help="evaluator-signed attestation JSON; required for prospective fits")
    args=cli.parse_args()
    try:
        if args.model_parameters is not None:
            check_model_parameter_target(args.model_parameters,args.deployed_parameters,
                                         label="standalone_model")
            print(json.dumps({"status":"PASS_100K_TARGET",
                              "trainable_parameters":args.model_parameters,
                              "deployed_parameters":args.deployed_parameters},sort_keys=True))
        elif args.manifest:
            r=check_manifest(
                json.loads(Path(args.manifest).read_text()), args.phase,
                reward_attestation=(
                    json.loads(Path(args.reward_evidence).read_text())
                    if args.reward_evidence else None),
                trusted_reward_public_key_hex=__import__("os").environ.get(
                    "TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX"))
            print(json.dumps(r,sort_keys=True))
        else:
            studies=check_changed(args.check_changed,Path.cwd())
            print(json.dumps({"status":"PASS","registered_changed_studies":studies},
                             sort_keys=True))
    except (ParityError,ValueError,KeyError,OSError,json.JSONDecodeError,subprocess.CalledProcessError) as e:
        print("RESEARCH_PARITY_HARD_FAILURE: "+str(e),file=sys.stderr)
        return 2
    return 0

if __name__=="__main__":
    sys.exit(main())
