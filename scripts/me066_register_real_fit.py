"""ME066 metadata-only untouched task registration and FIT-only scientific manifest.

The only publisher trajectories opened are original ME064 *training FIT* members
to recompute source order hashes. Both potential new heldout robot tasks are
examined from publisher META only; no CAL or future outcomes are opened.
The manifest is committed to Git BEFORE any trained optimizer update.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path

from scripts.me064_mixed_real_fit import prepare_fit, schedules
from scripts.me064_register_tasks import get_task
from scripts.me055_reserve_independent_robot_task import REVISION
from scripts.me060_train_trajectory import digest
from scripts.me065_prefit_original_fit import CONFIG, SEEDS

ROOT=Path("research/mathematical-engineering/me066")
ME065=Path("research/mathematical-engineering/me065/PREFLIGHT_MEASURED.json")
CANDIDATES=("plug_charger_into_power_strip",
            "replace_tissue_box_and_pull_out_first_tissue")
PREFIX=f"{REVISION}:ME066:NEW_UNTOUCHED_TASK:pre-train:v1:"
MIN_LENGTH=82

def sha(b: bytes)->str:
    return hashlib.sha256(b).hexdigest()

def generate(cache):
    measured=json.loads(ME065.read_text())
    if (not measured["numerical_parity_pass"] or
            measured["full_model_fits"]!=0 or
            measured["independent_holdout_scored"] or
            measured["original_real_fit_source_sha256"]!=
                "4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776"):
        raise RuntimeError("ME065 original resource preflight is not qualified")
    plan,identified,fit,source_hash,per_task_hashes=prepare_fit(cache)
    if source_hash!=measured["original_real_fit_source_sha256"]:
        raise RuntimeError("original source changed")
    if tuple(plan["remaining_untouched_tasks"])!=CANDIDATES:
        raise RuntimeError("unopened candidates differ from previous metadata registration")
    if set(CONFIG)!=set(measured["arms"]):
        raise RuntimeError("preflight architecture arms changed")
    heldout=min(CANDIDATES,key=lambda task:sha((PREFIX+task).encode()))
    metadata_cache=cache/"future-task-metadata-only"
    info,episodes,rows=get_task(heldout,metadata_cache)
    eligible=[(e,n) for e,n in rows if n>=MIN_LENGTH]
    by_block={}
    for ep,n in eligible:
        by_block.setdefault(ep//4,[]).append((ep,n))
    valid_blocks=[block for block,members in by_block.items() if len(members)==4]
    if len(valid_blocks)<20: raise RuntimeError("insufficient full untouched physical blocks")
    selected=sorted(valid_blocks,key=lambda block:(sha(
        f"{REVISION}:ME066:FROZEN-HOLDOUT-BLOCK:v1:{heldout}:{block}".encode()),block))[:20]
    cohort=sorted(ep for b in selected for ep in by_block[b])
    if len(cohort)!=80:raise RuntimeError("80 whole untouched episodes required")
    schedules_by_seed={str(seed):schedules(fit,seed,plan["training_tasks"]) for seed in SEEDS}
    source_cal_members={task:plan["task_evidence"][task]["CAL"]
                        for task in plan["training_tasks"]}
    splits={
        "FIT":source_hash,
        "CAL":digest(source_cal_members),
        "TARGET":digest(cohort),
    }
    registration={
        "study":"ME066_FOUR_ARM_REAL_ROBOT_PROSPECTIVE_TRAINING",
        "registered_before_fit":True,
        "original_robot":"r1_lite",
        "publisher_revision":REVISION,
        "training_task_names":plan["training_tasks"],
        "source_FIT_hash":source_hash,
        "training_source_task_sha256":per_task_hashes,
        "source_CAL_member_digest":splits["CAL"],
        "heldout_task_name":heldout,
        "heldout_meta_info_sha256":sha(info),
        "heldout_meta_episode_sha256":sha(episodes),
        "heldout_member_digest":splits["TARGET"],
        "heldout_original_episode_members":cohort,
        "heldout_original_trajectory_files_opened":0,
        "source_CAL_trajectory_files_opened":0,
        "exposed_previous_spray_nozzle_original_test_opened":0,
        "training_optimizer_updates":0,
        "heldout_selected_from_metadata_only":True,
        "other_untouched_task_reserved":[x for x in CANDIDATES if x!=heldout],
        "training_seed_sample_digests":{str(seed):digest(order)
                                         for seed,order in schedules_by_seed.items()},
        "source_FIT_episode_count":len(fit),
        "calibration":"source task CAL only",
        "limitations":"same robot source and exposed development training tasks; not independent site/operator; observational logged actions do not qualify counterfactual policy",
    }
    common_details={
        "dataset_sha256":source_hash,
        "train_split_sha256":splits["FIT"],
        "dev_split_sha256":splits["CAL"],
        "test_split_sha256":splits["TARGET"],
        "feature_contract_sha256":digest({"external_features":"same ME063 causal 14D state/14D action/authority/servo, recurrent history64"}),
        "target_contract_sha256":digest({"physical":"actual 14D physical state horizons 1..16 censored pre-human takeover"}),
        "optimizer_schedule_sha256":digest({"optimizer":"AdamW","lr":.001,"weight_decay":.0001,"clip":1.,"steps":600,"batch":8}),
        "tuning_policy_sha256":digest({"zero_tuning_trials":True,"fixed_geometries":CONFIG,"kl_rssm":.005}),
        "selection_policy_sha256":digest({"only_final_update":600,"no_new_target_access_before_freeze":True}),
        "evaluation_contract_sha256":digest({"holdout_task":heldout,"cohort":cohort,"per_horizon_and_motion":"frozen before target access"}),
        "hardware_profile":"GitHub Actions Linux CPU single torch thread, PyTorch 2.14.1+cpu",
    }
    max_flop=int(math.ceil(max(row["supported_operator_forward_backward_flops_per_update"]
                       for row in measured["arms"].values())*600*1.01))
    arms=[]
    for seed in SEEDS:
        for name,(width,decoder) in CONFIG.items():
            observed=measured["arms"][name]
            params=int(observed["trainable_parameters"])
            flop=int(observed["supported_operator_forward_backward_flops_per_update"])
            entry={
                "architecture":name,"seed":seed,
                "trainable_parameters":params,"deployed_parameters":params,
                "nontrainable_state_bytes":624 if name=="native_tac" else 720,
                "training_sample_order_sha256":digest(schedules_by_seed[str(seed)]),
                **common_details,
                "budget":{
                    "steps":600,"examples_seen":4800,"tokens_seen":384000,
                    "batch_size":8,"history_events":64,
                    "planned_training_flops":flop*600,
                    "max_training_flops":max_flop,
                    "tuning_trials":0,"tuning_flops":0,
                    "prefit_profile_method":"ME065 genuine original 160-episode five-task FIT, same real sample, PyTorch CPU with_flops on full 64+16 forward/backward, no optimizer or unsupported kernels",
                    "prefit_profile_training_flops":flop,
                }
            }
            arms.append(entry)
    manifest={
        "schema_version":"tac-osm.parity.v1",
        "study_id":"ME066_FOUR_ARM_GENUINE_ORIGINAL_ROBOT_FIT_EQUAL_BUDGET",
        "study_type":"architecture_comparison",
        "registered_before_first_fit":True,
        "real_world_source":{"license":"CC-BY-4.0 per BodenAI publisher card",
            "observed_real_outcomes":True,
            "name":"BodenAI RW-RL-HIL original five-task FIT original physical robot trajectories pinned rev "+REVISION,
            "sha256":source_hash},
        "literature_gap":{
            "unresolved_TAConly_question":"Does learned persistent fast/slow/associative native TAC memory, using unchanged physical features and priors, predict physical state better than compact RSSM, selective recurrent SSM, GRU and strongest persistence or servo on genuinely new actual robot task?",
            "what_is_not_being_rediscovered":"No novelty claim for RSSM, KL, recurrent GRU, input-selective stable SSM dynamics, Bayesian mixture-of-experts, servo physical forecasting or conformal uncertainty. Exact data/parameter/compute and real independent task outcomes are used as an empirical challenge.",
            "established_work":[
                {"paper":"Mastering Diverse Domains through World Models (DreamerV3)",
                 "established_result":"Recurrent deterministic and stochastic RSSM latent physical/visual world models, reward and imagined policy learning are known and much more complete than the compact RSSM control",
                 "url":"https://doi.org/10.1038/s41586-025-08744-2"},
                {"paper":"Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
                 "established_result":"Input-selective discretized state-space recurrent models are known; this micro diagonal control is not a complete Mamba reproduction",
                 "url":"https://arxiv.org/abs/2312.00752"}
            ]
        },
        "arms":arms,
        "prospective_untouched_target":{"task":heldout,
            "metadata_only":True,
            "cohort_digest":splits["TARGET"],"original_frozen_episodes":len(cohort)},
        "checkpoint_repo_owned_required":True,
        "training_authorized":False,
        "training_started":False,
        "source_fit_numeric_preflight":"research/mathematical-engineering/me065/PREFLIGHT_MEASURED.json",
        "full_hardware_flops_certified":False,
        "cpu_time_matching_required":False,
    }
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/"TASK_LEVEL_SPLIT.json").write_text(json.dumps(registration,indent=2,sort_keys=True)+"\n")
    (ROOT/"FAIR_COMPARISON.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print("ME066_METADATA_ONLY_PROSPECTIVE="+json.dumps({
        "task":heldout,"episodes":len(cohort),"fit_original_episodes":len(fit),
        "arm_count":len(arms),"new_target_trajectories_opened":0,
        "previous_exposed_test_reused":False},sort_keys=True),flush=True)

if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--source-cache",required=True,type=Path)
    a=cli.parse_args()
    generate(a.source_cache)
