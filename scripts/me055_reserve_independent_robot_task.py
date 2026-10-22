#!/usr/bin/env python3
"""Reserve a new real-robot task using metadata only; never open trajectories.

The task is selected by a revision-pinned hash before reading its episode
metadata. Split membership uses episode IDs and lengths only. Intervention
fields are deliberately ignored so the cohort cannot be selected for label
prevalence. This is source qualification, not permission to train.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

DATASET = "BodenAI/RW-RL-HIL-Dataset"
REVISION = "9290f5a96dc91ea58256e81d9ce2dd9621890b51"
USED_TASK = "scoop_cat_litter_into_trash_can"
CANDIDATES = (
    "clean_stains_off_tabletop",
    "fold_and_stack_scattered_clothes_neatly",
    "hang_clothes_on_drying_rack",
    "install_spray_nozzle_onto_bottle",
    "line_trash_can_with_trash_bag",
    "plug_charger_into_power_strip",
    "pour_water_from_kettle_to_make_tea",
    "replace_tissue_box_and_pull_out_first_tissue",
)
BLOCK = 4


class SourceFailure(ValueError):
    pass


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def chosen_task() -> str:
    return min(CANDIDATES, key=lambda task: sha(
        f"{REVISION}:independent-task:v1:{task}".encode()))


def download(task: str, name: str, maximum: int) -> bytes:
    url = (f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/" +
           quote(f"{task}/meta/{name}", safe="/"))
    with urlopen(Request(url, headers={"User-Agent": "TAC-OSM-source-audit/1.0"}),
                 timeout=90) as response:
        raw = response.read(maximum + 1)
    if len(raw) > maximum:
        raise SourceFailure("publisher metadata exceeded audit byte limit")
    return raw


def metadata_rows(raw: bytes):
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        episode, length = source.get("episode_index"), source.get("length")
        if (type(episode) is not int or episode < 0 or
                type(length) is not int or length < 18):
            raise SourceFailure("invalid publisher episode ID/length")
        # Do not read intervention, control, reward or success values here.
        rows.append((episode, length))
    rows.sort()
    if [episode for episode, _ in rows] != list(range(len(rows))):
        raise SourceFailure("publisher episode IDs are not unique and contiguous")
    return rows


def reserve(rows):
    blocks = sorted({episode // BLOCK for episode, _ in rows}, key=lambda block:
                    (sha(f"{REVISION}:{chosen_task()}:me055:block:{block}".encode()), block))
    n = len(blocks)
    train_end, dev_end = int(n * .60), int(n * .80)
    if train_end < 1 or dev_end <= train_end or dev_end >= n:
        raise SourceFailure("not enough source blocks for three cohorts")
    ownership = {
        "FIT": set(blocks[:train_end]),
        "DEV": set(blocks[train_end:dev_end]),
        "TEST": set(blocks[dev_end:]),
    }
    split = {name: [(i, length) for i, length in rows if i // BLOCK in keep]
             for name, keep in ownership.items()}
    flat = [i for part in split.values() for i, _ in part]
    if sorted(flat) != list(range(len(rows))) or len(flat) != len(set(flat)):
        raise SourceFailure("metadata-only cohort overlap or omission")
    return split


def build(info_raw: bytes, episodes_raw: bytes):
    task = chosen_task()
    info = json.loads(info_raw)
    rows = metadata_rows(episodes_raw)
    if (info.get("robot_type") != "r1_lite" or info.get("fps") != 15 or
            info.get("total_episodes") != len(rows) or
            info.get("total_frames") != sum(length for _, length in rows)):
        raise SourceFailure("publisher task metadata is internally inconsistent")
    features = info.get("features", {})
    for name in ("observations.state.qpos", "action.qpos"):
        if features.get(name, {}).get("shape") != [14]:
            raise SourceFailure("independent task lacks the registered 14D interface")
    split = reserve(rows)
    receipt = {
        "schema": "tac-osm.me055.metadata-reservation.v1",
        "dataset": DATASET,
        "immutable_revision": REVISION,
        "task_selection_rule": "minimum SHA256(revision:independent-task:v1:task)",
        "candidate_tasks_sorted": sorted(CANDIDATES),
        "selected_task": task,
        "selection_made_without_episode_metadata_or_outcomes": True,
        "publisher_info_sha256": sha(info_raw),
        "publisher_episodes_metadata_sha256": sha(episodes_raw),
        "robot_type": info["robot_type"],
        "fps": info["fps"],
        "episode_count": len(rows),
        "frame_count_from_lengths": sum(length for _, length in rows),
        "trajectory_files_opened": 0,
        "intervention_or_reward_values_parsed": False,
        "optimizer_steps": 0,
        "cohorts": {},
        "training_authorized": False,
        "remaining_gates": [
            "FIT trajectory checksum and schema audit",
            "multi-step target and censoring contract",
            "parameter, data, FLOP and CPU preflight parity",
            "frozen architecture and five-seed training protocol",
        ],
    }
    for name, members in split.items():
        ids = [episode for episode, _ in members]
        receipt["cohorts"][name] = {
            "episode_count": len(ids),
            "frame_count_from_metadata": sum(length for _, length in members),
            "episode_id_sha256": sha(json.dumps(ids, separators=(",", ":")).encode()),
            "minimum_episode_length": min(length for _, length in members),
        }
    return receipt


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", required=True)
    args = cli.parse_args()
    task = chosen_task()
    try:
        receipt = build(download(task, "info.json", 1_000_000),
                        download(task, "episodes.jsonl", 2_000_000))
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        print(json.dumps({"event": "ME055_INDEPENDENT_TASK_METADATA_RESERVED",
                          "task": task, "episodes": receipt["episode_count"],
                          "trajectory_files_opened": 0}))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print("ME055_SOURCE_RESERVATION_FAILURE: " + str(error),
              file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
