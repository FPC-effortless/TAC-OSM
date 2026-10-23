"""ME062 prospectively reserve a DIFFERENT authentic robot task via metadata ONLY.

No target parquet, actions, state outcomes, human intervention, or fitted-model
scores are read. This preregistration is not a declaration of independent
operator/site conditions or a completed transfer result.
"""
import hashlib
import json
from pathlib import Path

from scripts.audit_real_hil_trajectory import download_file
from scripts.me055_reserve_independent_robot_task import (
    REVISION, CANDIDATES, metadata_rows,
)
HISTORY, HORIZON = 64, 16


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2))


ROOT = Path("research/mathematical-engineering/me062")
EXCLUDE = (
    "clean_stains_off_tabletop",       # source historically exposed
    "hang_clothes_on_drying_rack",    # ME060 transfer exposed
    "pour_water_from_kettle_to_make_tea", # ME061 transfer exposed
)
TASKS = tuple(sorted(set(CANDIDATES) - set(EXCLUDE)))


def choose():
    return min(TASKS, key=lambda task: hashlib.sha256(
        f"{REVISION}:me062-transfer:v1:{task}".encode()).hexdigest())


def reserve(cache):
    task = choose()
    info_raw = download_file(REVISION, f"{task}/meta/info.json",
                             cache/task/"meta/info.json", 1_000_000)
    episodes_raw = download_file(REVISION, f"{task}/meta/episodes.jsonl",
                                 cache/task/"meta/episodes.jsonl", 2_000_000)
    info = json.loads(info_raw)
    rows = metadata_rows(episodes_raw)
    if (info.get("robot_type") != "r1_lite" or info.get("fps") != 15 or
            info.get("total_episodes") != len(rows) or
            info.get("total_frames") != sum(n for _, n in rows)):
        raise RuntimeError("target metadata does not match frozen robot interface")
    for name in ("observations.state.qpos", "action.qpos"):
        if info.get("features", {}).get(name, {}).get("shape") != [14]:
            raise RuntimeError("target lacks registered 14D physical feature interface")
    blocks = sorted(
        {episode//4 for episode,length in rows if length > HISTORY+HORIZON},
        key=lambda block: (digest([task, "me062-frozen-block", block]), block))
    if len(blocks) < 20:
        raise RuntimeError("target has too few authentic eligible episode blocks")
    kept = set(blocks[:20])
    cohort = [(e,n) for e,n in rows if e//4 in kept and n > HISTORY+HORIZON]
    if len(cohort) < 40:
        raise RuntimeError("insufficient physical target episodes")
    registration = {
        "study": "ME062_FROZEN_DIFFERENT_TASK_TRANSFER",
        "task": task, "revision": REVISION,
        "available_tasks": list(TASKS),
        "excluded_exposed_tasks": list(EXCLUDE),
        "selection": "minimum SHA256(revision:me062-transfer:v1:task) before episode metadata/outcome access",
        "info_sha256": sha_bytes(info_raw),
        "episodes_metadata_sha256": sha_bytes(episodes_raw),
        "cohort": cohort, "cohort_sha256": digest(cohort),
        "robot_type": "r1_lite", "fps": 15,
        "training_updates": 0,
        "target_trajectory_files_opened": 0,
        "target_outcome_data_opened": False,
        "checkpoint_selection": "all ten registered original source final checkpoints frozen before transfer; no target fitting, preprocessing or calibration",
        "evaluation": "identical source physical14 adapter, midpoint robot-control origin, 64 factual history transitions, 16 logged candidate factual action prefix, censor at first human takeover",
        "acceptance": "native TAC >=2% lower real physical MAE than each trained GRU/persistence/servo, episode-paired interval upper bound below zero; uncertainty/risk-coverage not degraded",
        "scope": "different real recorded physical task on same robot interface; site/operator independence unknown, no counterfactual action value",
        "original_test_independence": "NOT_CLAIMED_NEITHER_OPERATOR_NOR_SITE_CERTIFIED",
        "requires_prior_repo_checkpoint_freeze": True,
        "scientific_status": "REGISTERED_METADATA_ONLY_NO_TRANSFER_SCORING",
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(ROOT/"TRANSFER_PROTOCOL.json", registration)
    print("ME062_TARGET_METADATA_ONLY="+json.dumps({
        "task":task,"episodes":len(cohort),"outcomes_opened":0,
        "metadata_sha256":registration["episodes_metadata_sha256"]}),flush=True)


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-cache",type=Path,required=True)
    a=parser.parse_args()
    reserve(a.source_cache)
