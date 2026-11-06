"""No synthetic trajectory is used for model performance; fixture-only contracts."""
import json
import pytest

from scripts.audit_real_hil_trajectory import (
    SourceFailure, audit_episodes, audit_frame_table, EXPECTED_ROBOT_DIM)

class FakeArray:
    def __init__(self, values):
        self.values=values
    def to_pylist(self):
        return self.values

class FakeTable:
    def __init__(self, data):
        self.data=data
        self.column_names=list(data)
        self.num_rows=len(next(iter(data.values())))
    def __getitem__(self, name):
        return FakeArray(self.data[name])
    def set_column(self, i, name, values):
        fresh=dict(self.data)
        fresh[name]=values
        return FakeTable(fresh)
    def append_column(self, name, values):
        fresh=dict(self.data)
        fresh[name]=values
        return FakeTable(fresh)

def example_episodes(length=5):
    return (json.dumps({"episode_index":0,"length":length,
        "has_intervention":True,"intervention_frames":2,
        "intervention_ranges":[{"segment_index":0,"start_frame":1,"end_frame":2}]})+
        "\n"+json.dumps({"episode_index":1,"length":4,
        "has_intervention":False,"intervention_frames":0,"intervention_ranges":[]})+"\n").encode()

def table_episode(mask=(False,True,True,False,False), bad_times=False):
    n=len(mask)
    return FakeTable({
        "timestamp":[i/15 if not bad_times else 0.0 for i in range(n)],
        "frame_index":list(range(n)),
        "episode_index":[0]*n,
        "intervention":list(mask),
        "control_source":["intervention" if v else "policy" for v in mask],
        "action.qpos":[[float(i)]*EXPECTED_ROBOT_DIM for i in range(n)],
        "observations.state.qpos":[[float(i)+.1]*EXPECTED_ROBOT_DIM for i in range(n)]
    })

def test_source_aggregate_is_not_reward_or_policy_success_label():
    summary=audit_episodes(example_episodes())
    assert summary["episode_count"]==2
    assert summary["episodes_with_human_intervention"]==1
    assert summary["episodes_without_human_intervention"]==1
    detail=audit_frame_table(table_episode(),{"length":5,"intervention_frames":2,"episode_index":0})
    assert detail["measured_state_dimensions"]==detail["executed_action_dimensions"]==14
    assert detail["explicit_scalar_reward_done_success_columns"]==[]
    assert detail["human_control_frames"]==2

def test_no_missing_real_episode_indices_or_intervention_ranges_allowed():
    event=json.loads(example_episodes().splitlines()[0])
    event["episode_index"]=2
    with pytest.raises(SourceFailure,match="contiguous"):
        audit_episodes((json.dumps(event)+"\n").encode())
    event["episode_index"]=0
    event["intervention_frames"]=1
    with pytest.raises(SourceFailure,match="disagree"):
        audit_episodes((json.dumps(event)+"\n").encode())

def test_missing_reward_cannot_be_silently_invented():
    table=table_episode()
    detail=audit_frame_table(table,{"length":5,"intervention_frames":2,"episode_index":0})
    assert not detail["explicit_scalar_reward_done_success_columns"]
    enriched=table.append_column("reward",[0.,0.,0.,0.,0.])
    assert audit_frame_table(enriched,{"length":5,"intervention_frames":2,"episode_index":0})[
        "explicit_scalar_reward_done_success_columns"]==["reward"]

def test_unknown_control_authority_invalidates_real_transition_labels():
    table=table_episode().set_column(4,"control_source",["policy"]*5)
    with pytest.raises(SourceFailure,match="ownership"):
        audit_frame_table(table,{"length":5,"intervention_frames":2,"episode_index":0})

def test_time_and_state_dimensional_contracts_fail_closed():
    with pytest.raises(SourceFailure,match="chronological"):
        audit_frame_table(table_episode(bad_times=True),
            {"length":5,"intervention_frames":2,"episode_index":0})
    bad=table_episode().set_column(5,"action.qpos",[[1.,2.]]*5)
    with pytest.raises(SourceFailure,match="14 finite"):
        audit_frame_table(bad,{"length":5,"intervention_frames":2,"episode_index":0})

def test_original_metadata_and_parquet_frame_counts_must_match():
    with pytest.raises(SourceFailure,match="match metadata"):
        audit_frame_table(table_episode(),{"length":9,"intervention_frames":2,"episode_index":0})
