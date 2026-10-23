"""ME064 task-level FIT/CAL and genuinely untouched holdout registration.

Publisher metadata only: no parquet, intervention/reward/physical outcome parsing.
Frozen train task list is historically development-exposed, not independent.
"""
from __future__ import annotations
import hashlib, json, argparse
from pathlib import Path

from scripts.audit_real_hil_trajectory import download_file
from scripts.me055_reserve_independent_robot_task import REVISION, metadata_rows, reserve

TRAIN_TASKS = (
    "clean_stains_off_tabletop",
    "fold_and_stack_scattered_clothes_neatly",
    "hang_clothes_on_drying_rack",
    "line_trash_can_with_trash_bag",
    "pour_water_from_kettle_to_make_tea",
)
# No previously used task is eligible for the untouched task-level test.
UNTOUCHED_CANDIDATES = (
    "install_spray_nozzle_onto_bottle",
    "plug_charger_into_power_strip",
    "replace_tissue_box_and_pull_out_first_tissue",
)
TRAIN_PER_TASK, CAL_PER_TASK = 32, 8
MINIMUM_LENGTH = 82
ROOT = Path("research/mathematical-engineering/me064")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def select_holdout():
    return min(UNTOUCHED_CANDIDATES,
               key=lambda t: sha(f"{REVISION}:ME064:UNTOUCHED_TASK:v1:{t}".encode()))


def get_task(task, cache):
    info_bytes = download_file(REVISION, f"{task}/meta/info.json",
             cache/task/"meta/info.json", 1_000_000)
    episodes_bytes = download_file(REVISION, f"{task}/meta/episodes.jsonl",
             cache/task/"meta/episodes.jsonl", 2_000_000)
    info=json.loads(info_bytes)
    rows=metadata_rows(episodes_bytes)
    if (info.get("robot_type")!="r1_lite" or info.get("fps")!=15 or
        info.get("total_episodes")!=len(rows) or
        info.get("total_frames")!=sum(n for _,n in rows)):
        raise RuntimeError("published task metadata inconsistent: "+task)
    for f in ("observations.state.qpos","action.qpos"):
        if info.get("features",{}).get(f,{}).get("shape") != [14]:
            raise RuntimeError("incompatible real original robot channels: "+task)
    return info_bytes, episodes_bytes, rows


def eligible_training(task, rows):
    # Preserve legacy source original TEST exclusion for clean-stains source.
    if task == "clean_stains_off_tabletop":
        allowed = reserve(rows)["FIT"]
        return [p for p in allowed if p[1] >= MINIMUM_LENGTH]
    return [p for p in rows if p[1] >= MINIMUM_LENGTH]


def select_blocks(task, rows):
    eligible=eligible_training(task,rows)
    by_block={}
    for ep,n in eligible:
        by_block.setdefault(ep//4,[]).append((ep,n))
    full=[b for b,episodes in by_block.items() if len(episodes)==4]
    ordered=sorted(full,key=lambda b:(sha(f"{REVISION}:ME064:FROZEN_SPLIT:v1:{task}:{b}".encode()),b))
    if len(ordered)<10:
        raise RuntimeError("fewer than 10 complete 4-episode eligible blocks: "+task)
    fit_blocks=set(ordered[:8])
    cal_blocks=set(ordered[8:10])
    fit=sorted((ep,n) for block in fit_blocks for ep,n in by_block[block])
    cal=sorted((ep,n) for block in cal_blocks for ep,n in by_block[block])
    if len(fit)!=TRAIN_PER_TASK or len(cal)!=CAL_PER_TASK:
        raise RuntimeError("cross-task 32/8 episode balance failed: "+task)
    return fit,cal


def register(cache):
    heldout=select_holdout()
    if heldout in TRAIN_TASKS: raise RuntimeError("holdout leaked into training")
    task_evidence={}
    for task in (*TRAIN_TASKS,heldout):
        info,episodes,rows=get_task(task,cache)
        receipt={"info_sha256":sha(info),"episodes_metadata_sha256":sha(episodes),
                 "total_original_episodes":len(rows),"robot_type":"r1_lite",
                 "fps":15,"original_trajectory_files_opened":0}
        if task in TRAIN_TASKS:
            fit,cal=select_blocks(task,rows)
            receipt.update({"FIT":fit,"CAL":cal,
                "FIT_episode_sha256":digest(fit),"CAL_episode_sha256":digest(cal)})
        else:
            # One metadata-only 80-episode evaluation cohort, 4-episode
            # recording blocks; no selection based on interventions/outcomes.
            eligible=[(e,n) for e,n in rows if n >= MINIMUM_LENGTH]
            blocks={e//4 for e,n in eligible}
            order=sorted(blocks,key=lambda b:(sha(
                 f"{REVISION}:ME064:HELDOUT:v1:{task}:{b}".encode()),b))
            eligible_blocks=set(order[:20])
            cohort=sorted((e,n) for e,n in eligible if e//4 in eligible_blocks)
            if len(cohort)<40:
                raise RuntimeError("insufficient untouched physical target episodes")
            receipt.update({"FROZEN_HOLDOUT":cohort,
                "holdout_episode_sha256":digest(cohort),
                "selection_rule":"metadata-only hash-ordered 20 recording blocks"})
        task_evidence[task]=receipt
    plan={
        "study":"ME064_MULTI_TASK_LEAVE_ONE_TASK_OUT_REAL_WORLD",
        "source":"BodenAI/RW-RL-HIL-Dataset",
        "immutable_revision":REVISION,"original_14D_robot":"r1_lite","fps":15,
        "training_tasks":list(TRAIN_TASKS),"holdout_task":heldout,
        "remaining_untouched_tasks":[x for x in UNTOUCHED_CANDIDATES if x!=heldout],
        "training_episode_count":sum(len(task_evidence[t]["FIT"]) for t in TRAIN_TASKS),
        "cal_episode_count":sum(len(task_evidence[t]["CAL"]) for t in TRAIN_TASKS),
        "holdout_episode_count":len(task_evidence[heldout]["FROZEN_HOLDOUT"]),
        "task_evidence":task_evidence,
        "model_selection":"fixed final update 600 only; five new paired seeds",
        "task_sampling":"roundrobin task index each FIT optimizer update, then uniform with replacement original FIT eligible origins of selected task",
        "normalization_and_servo":"joint FIT ONLY, never source CAL or held-out task",
        "independence_limit":"training tasks development-exposed historically; single unseen task holdout, same robot platform; no verified independent site/operator",
        "calibration":"source training task CAL only, never heldout target",
        "ineligible_claims":["multisite generalization","causal intervention value","previous exposed task independence","calibrated autonomous decision"],
        "train_trajectory_files_opened":0,
        "calibration_trajectory_files_opened":0,
        "holdout_trajectory_files_opened":0,
        "training_updates":0,
    }
    ROOT.mkdir(parents=True,exist_ok=True)
    output=ROOT/"TASK_LEVEL_SPLIT.json"
    output.write_text(json.dumps(plan,sort_keys=True,indent=2)+"\n")
    print("ME064_METADATA_ONLY_REGISTERED "+json.dumps({
        "training_tasks":list(TRAIN_TASKS),"holdout":heldout,
        "train_episodes":plan["training_episode_count"],"cal_episodes":plan["cal_episode_count"],
        "heldout_episodes":plan["holdout_episode_count"],"trajectory_files_opened":0}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--cache",type=Path,required=True)
    args=p.parse_args()
    register(args.cache)
