"""ME066 fail-closed authorization after the original real FIT science and Git storage evidence.

Writes the final source manifest permission and provenance receipt BEFORE
any optimizer training. It does not read source CAL, heldout robot trajectories,
make predictions, or fit even one model.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path

from scripts.validate_research_parity import check_manifest

ROOT=Path("research/mathematical-engineering/me066")
OUT=Path("research/replay-bundles/ME066")
SOURCE_PREFIT=Path("research/mathematical-engineering/me065/PREFLIGHT_MEASURED.json")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    fair=ROOT/"FAIR_COMPARISON.json"
    target=ROOT/"TASK_LEVEL_SPLIT.json"
    permission=ROOT/"TRAINING_PERMISSION.json"
    probe=OUT/"STORAGE_PROBE_MANIFEST.json"
    binary=OUT/"UNTRAINED_STORAGE_PROBE.pt"
    if permission.exists():
        raise RuntimeError("original training permission already frozen; no second authorization")
    if (OUT/"TRAINING_RESULT.json").exists() or (OUT/"CHECKPOINTS_FROZEN_PARTIAL.json").exists():
        raise RuntimeError("fit has already started; cannot retrospectively preregister")
    for p in (fair,target,probe,binary,SOURCE_PREFIT):
        if not p.is_file():
            raise RuntimeError("pre-training original Git research prerequisite absent: "+str(p))
    m=json.loads(fair.read_text())
    r=json.loads(target.read_text())
    p=json.loads(probe.read_text())
    numerical=json.loads(SOURCE_PREFIT.read_text())
    if m["training_authorized"] or not m["registered_before_first_fit"]:
        raise RuntimeError("original FIT preregistration already used or absent")
    if not numerical["numerical_parity_pass"] or not numerical["all_active"]:
        raise RuntimeError("source FIT architecture parity originally failed")
    if numerical["full_model_fits"]!=0 or numerical["independent_holdout_scored"]:
        raise RuntimeError("preflight already trained or revealed outcome")
    if len(m["arms"])!=20 or len({row["seed"] for row in m["arms"]})!=5 or len({
            row["architecture"] for row in m["arms"]})!=4:
        raise RuntimeError("complete four by five prospective fitted matrix not registered")
    if (r["source_FIT_hash"]!=m["real_world_source"]["sha256"] or
            r["heldout_original_trajectory_files_opened"]!=0 or
            r["source_CAL_trajectory_files_opened"]!=0 or
            r["exposed_previous_spray_nozzle_original_test_opened"]!=0 or
            r["training_optimizer_updates"]!=0):
        raise RuntimeError("not source-only future metadata or source original FIT")
    if (r["heldout_task_name"]!="plug_charger_into_power_strip" or
            r["heldout_original_episode_members"] is None or
            len(r["heldout_original_episode_members"])!=80):
        raise RuntimeError("unknown untouched registered physical robot task")
    if p["status"]!="GIT_STORAGE_PROBE_PASS" or sha(binary)!=p["sha256"] or p["trained_model_count"]!=0:
        raise RuntimeError("Git-owned original binary checkpoint route not proven")
    if check_manifest(m,"preflight")["status"]!="PASS":
        raise RuntimeError("original 20-arm numeric FAIR validator failed")
    for row in m["arms"]:
        measured=numerical["arms"].get(row["architecture"])
        if not measured or (row["trainable_parameters"]!=measured["trainable_parameters"] or
                row["budget"]["prefit_profile_training_flops"]!=measured[
                    "supported_operator_forward_backward_flops_per_update"]):
            raise RuntimeError("prospective 20-arm FIT numeric matched budget altered")
        if row["training_sample_order_sha256"]!=r["training_seed_sample_digests"][str(row["seed"])]:
            raise RuntimeError("prospective paired seed real FIT source schedule altered")
    frozen_pre_authorization=sha(fair)
    m["training_authorized"]=True
    m["training_authority"]="owner-directed current research continuation; original FIT only, no heldout scoring"
    m["original_pre_authorization_manifest_sha256"]=frozen_pre_authorization
    fair.write_text(json.dumps(m,indent=2,sort_keys=True)+"\n")
    if check_manifest(m,"preflight")["status"]!="PASS":
        raise RuntimeError("same original future source parity failed after permission")
    receipt={
        "study":m["study_id"],
        "train_original_FIT_only":True,
        "source_manifest_sha256":sha(fair),
        "source_manifest_before_authorization_sha256":frozen_pre_authorization,
        "source_FIT_original_digest":r["source_FIT_hash"],
        "source_FIT_episode_count":160,
        "new_untouched_robot_task":r["heldout_task_name"],
        "source_CAL_read_allowed_before_checkpoint_freeze":False,
        "target_trajectory_access_allowed":False,
        "old_exposed_target_reuse":False,
        "git_owned_original_checkpoint_probe_sha256":p["sha256"],
        "trained_model_count_before_permission":0,
        "paired_seed_count":5,
        "registered_architecture_arm_count":4,
        "training_update_budget_per_arm_seed":600,
        "full_optimizer_and_hardware_flops_certified":False,
        "cpu_runtime_equality_compulsory":False,
        "status":"FIT_TRAINING_ONLY_AUTHORIZED_NO_MODEL_YET",
    }
    permission.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print("ME066_REAL_ORIGINAL_FIT_TRAINING_AUTHORIZED="+json.dumps({
        "arm_count":4,"seed_pairs":5,"optimizer_updates_completed":0,
        "new_untouched_task":r["heldout_task_name"],
        "source_manifest_sha256":receipt["source_manifest_sha256"]},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
