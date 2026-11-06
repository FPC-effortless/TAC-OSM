"""Contracts use constructed fixtures ONLY; no synthetic performance evidence."""
import json
import pytest

from scripts.qualify_real_robot_cohort import (
    metadata_only_rows, partition_metadata, cohort_receipt, deterministic_fit_sample,
    SourceFailure, REVISION, EXPECTED_EPISODES)
from tac_osm.real_robot_transition_contract import (
    real_windows_from_original_episode,assert_no_future_inputs)


class FakeArrow:
    def __init__(self,values):self.values=values
    def to_pylist(self):return self.values


class FakeTable:
    def __init__(self,n=20,next_human_at=9):
        self.num_rows=n
        self.data={
            "frame_index":list(range(n)),
            "episode_index":[7]*n,
            "timestamp":[i/15 for i in range(n)],
            "intervention":[i==next_human_at for i in range(n)],
            "control_source":["intervention" if i==next_human_at else "policy" for i in range(n)],
            "observations.state.qpos":[[float(i+j/100) for j in range(14)] for i in range(n)],
            "action.qpos":[[float(i+j/1000) for j in range(14)] for i in range(n)],
        }
    def __getitem__(self,key):
        return FakeArrow(self.data[key])


def metadata_fixture(human_case=False):
    # In a contract-only fixture, no real robot records or outcomes are generated
    # or scored; labels are never inputs to the cohort partitioner.
    rows=[]
    for i in range(EXPECTED_EPISODES):
        rows.append({"episode_index":i,"length":270+i%70,
                     "has_intervention":bool((i%2)==0) if human_case else False,
                     "intervention_ranges":[{"start_frame":20,"end_frame":25}] if human_case else []})
    return ("\n".join(json.dumps(r) for r in rows)+"\n").encode()


def test_real_robot_cohorts_are_complete_disjoint_and_reproducible():
    rows=metadata_only_rows(metadata_fixture())
    a=partition_metadata(rows)
    b=partition_metadata(rows)
    assert a==b
    assert [len(a[k]) for k in ("FIT","DEV","TEST")]==[325,80,100]
    assert all(set(i//5 for i,_ in a[k]).isdisjoint(
        i//5 for i,_ in a[j]) for k in a for j in a if k!=j)
    assert len(deterministic_fit_sample(a))==20
    receipt=cohort_receipt(rows,a,{"publisher_sha":"0"*64})
    assert receipt["number_of_models_trained"]==0
    assert receipt["strictly_sealed_dev_and_test_parquet"]
    assert receipt["splits"]["TEST"]["eligible_real_one_step_pairs_by_prior_history"]["256"]>0


def test_tampering_only_intervention_labels_does_not_change_any_episode_split():
    a=partition_metadata(metadata_only_rows(metadata_fixture(False)))
    b=partition_metadata(metadata_only_rows(metadata_fixture(True)))
    assert a==b


def test_metadata_missing_episodes_and_duplicate_ids_are_hard_errors():
    raw=metadata_fixture()
    with pytest.raises(SourceFailure,match="count"):
        metadata_only_rows(b"\n".join(raw.splitlines()[:-1]))
    sample=[json.loads(x) for x in raw.splitlines()]
    sample[9]["episode_index"]=8
    changed=("\n".join(json.dumps(x) for x in sample)).encode()
    with pytest.raises(SourceFailure,match="duplicate"):
        metadata_only_rows(changed)


def test_genuine_temporal_contract_is_predecision_and_never_reads_next_state_as_input():
    table=FakeTable()
    samples=list(real_windows_from_original_episode(table,7,history=8))
    assert len(samples)==12
    s=next(x for x in samples if x.current_frame_index==8)
    assert len(s.observed_history_state)==8
    assert len(s.previously_executed_actions)==7
    assert s.current_logged_action==tuple(table.data["action.qpos"][8])
    assert s.current_measured_state==tuple(table.data["observations.state.qpos"][8])
    assert s.factual_targets()["next_measured_state"]==tuple(table.data["observations.state.qpos"][9])
    assert s.factual_targets()["next_intervention_onset_when_policy_controlled"] is True
    assert assert_no_future_inputs(s)


def test_future_label_mutation_only_changes_supervised_targets_not_predecision_inputs():
    original=FakeTable()
    changed=FakeTable()
    changed.data["observations.state.qpos"][9]=[999.]*14
    changed.data["intervention"][9]=False
    changed.data["control_source"][9]="policy"
    a=next(s for s in real_windows_from_original_episode(original,7,8)
           if s.current_frame_index==8)
    b=next(s for s in real_windows_from_original_episode(changed,7,8)
           if s.current_frame_index==8)
    assert a.model_inputs()==b.model_inputs()
    assert a.factual_targets()!=b.factual_targets()


def test_real_action_policy_human_authority_and_episode_boundaries_are_preserved():
    samples=list(real_windows_from_original_episode(FakeTable(),7,8))
    while_human=next(s for s in samples if s.current_frame_index==9)
    assert while_human.currently_human_controlled
    assert while_human.factual_targets()["next_intervention_onset_when_policy_controlled"] is None
    assert all(s.episode_index==7 for s in samples)
    fake=FakeTable()
    fake.data["episode_index"][8]=8
    with pytest.raises(ValueError,match="continuity"):
        list(real_windows_from_original_episode(fake,7,8))


def test_real_frame_h256_pair_count_matches_manifest_algebra_and_no_leakage():
    table=FakeTable(n=260,next_human_at=257)
    samples=list(real_windows_from_original_episode(table,7,history=256))
    assert len(samples)==4
    assert samples[0].current_frame_index==255
    assert samples[-1].current_frame_index==258
    assert all(assert_no_future_inputs(s) for s in samples)


def test_invalid_future_timestamp_or_actions_abort_instead_of_imputing():
    bad=FakeTable()
    bad.data["timestamp"][5]=bad.data["timestamp"][4]
    with pytest.raises(ValueError,match="noncausal"):
        list(real_windows_from_original_episode(bad,7,8))
    bad=FakeTable()
    bad.data["action.qpos"][4]=[float("nan")]*14
    with pytest.raises(ValueError,match="nonfinite"):
        list(real_windows_from_original_episode(bad,7,8))
