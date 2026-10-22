"""Generated metadata fixtures test the source gate, never model capability."""
import json
import pytest
from scripts.me055_reserve_independent_robot_task import (
    CANDIDATES, REVISION, SourceFailure, build, chosen_task, sha)


def fixture(n=40):
    info = {
        "robot_type": "r1_lite", "fps": 15, "total_episodes": n,
        "total_frames": sum(100 + i for i in range(n)),
        "features": {"observations.state.qpos": {"shape": [14]},
                     "action.qpos": {"shape": [14]}},
    }
    episodes = b"".join((json.dumps({"episode_index": i, "length": 100+i,
                                      "intervention_frames": 999}).encode()+b"\n")
                         for i in range(n))
    return json.dumps(info).encode(), episodes


def test_task_selected_by_frozen_rule_before_metadata():
    expected = min(CANDIDATES, key=lambda task: sha(
        f"{REVISION}:independent-task:v1:{task}".encode()))
    assert chosen_task() == expected == "clean_stains_off_tabletop"


def test_reservation_is_disjoint_and_does_not_parse_outcome_fields():
    info, episodes = fixture()
    receipt = build(info, episodes)
    assert receipt["episode_count"] == 40
    assert receipt["trajectory_files_opened"] == 0
    assert not receipt["intervention_or_reward_values_parsed"]
    assert not receipt["training_authorized"]
    assert sum(x["episode_count"] for x in receipt["cohorts"].values()) == 40


@pytest.mark.parametrize("change", ["robot", "fps", "state", "action", "count"])
def test_source_contract_fails_closed(change):
    info_raw, episodes = fixture()
    info = json.loads(info_raw)
    if change == "robot": info["robot_type"] = "unknown"
    elif change == "fps": info["fps"] = 10
    elif change == "state": info["features"]["observations.state.qpos"]["shape"] = [13]
    elif change == "action": info["features"]["action.qpos"]["shape"] = [15]
    else: info["total_episodes"] += 1
    with pytest.raises(SourceFailure):
        build(json.dumps(info).encode(), episodes)
