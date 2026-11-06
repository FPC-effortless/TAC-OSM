"""Software contract tests only: generated tables are NEVER intelligence evidence."""
import pytest

from scripts.audit_full_real_robot_fit import (
    audit_factual_fit_trajectory,build_full_fit_receipt,source_path,SCHEMA)
from scripts.qualify_real_robot_cohort import (
    metadata_only_rows,partition_metadata,cohort_receipt,EXPECTED_EPISODES)
from scripts.audit_real_hil_trajectory import SourceFailure


class DataArray:
    def __init__(self,data):self.data=data
    def to_pylist(self):return self.data


class TableFixture:
    def __init__(self,n,episode_id,human=(30,31,100)):
        self.num_rows=n
        human=set(human)
        self.data={
            "timestamp":[i/15. for i in range(n)],
            "frame_index":list(range(n)),
            "episode_index":[episode_id]*n,
            "intervention":[i in human for i in range(n)],
            "control_source":["intervention" if i in human else "policy" for i in range(n)],
            "action.qpos":[[float(i)]*14 for i in range(n)],
            "observations.state.qpos":[[float(i)+0.2]*14 for i in range(n)]
        }
    def __getitem__(self,key):return DataArray(self.data[key])
    @property
    def column_names(self):return list(self.data)


def one_metadata(length,episode_id=4):
    return {"episode_index":episode_id,"length":length,"intervention_frames":3,
            "intervention_ranges":[{"start_frame":30,"end_frame":31},
                                   {"start_frame":100,"end_frame":100}],
            "has_intervention":True}


def all_original_shape_fixture():
    import json
    return ("\n".join(json.dumps({"episode_index":i,"length":300})
                      for i in range(EXPECTED_EPISODES))+"\n").encode()


def test_real_factual_fit_support_does_not_mislabel_existing_human_control_as_new_onset():
    report=audit_factual_fit_trajectory(TableFixture(300,4),4,one_metadata(300))
    assert report["frames"]==300
    assert report["human_frames"]==3
    assert report["takeover_onsets_all"]==2
    h=report["histories"]["256"]
    assert h["real_observed_state_action_next_state_pairs"]==44
    assert h["operator_policy_eligible_onset_examples"]==44
    assert h["actual_human_takeover_onset_positives"]==0
    assert h["masked_already_human_frames"]==0


def test_actual_takeover_boundary_positive_at_h256():
    t=TableFixture(300,4,human=(260,261,299))
    meta={"episode_index":4,"length":300,"intervention_frames":3,
          "intervention_ranges":[{"start_frame":260,"end_frame":261},
                                  {"start_frame":299,"end_frame":299}],
          "has_intervention":True}
    s=audit_factual_fit_trajectory(t,4,meta)
    assert s["histories"]["256"]["actual_human_takeover_onset_positives"]==2
    assert s["histories"]["256"]["masked_already_human_frames"]==2


def test_full_fit_source_receipt_requires_every_original_325_episodes():
    raw=all_original_shape_fixture()
    parts=partition_metadata(metadata_only_rows(raw))
    one=audit_factual_fit_trajectory(TableFixture(300,4),4,one_metadata(300))
    details={i:{**one,"frames":300} for i,_ in parts["FIT"]}
    digests={i:"c"*64 for i,_ in parts["FIT"]}
    r=build_full_fit_receipt({"meta":"d"*64},parts,digests,details)
    assert r["schema_version"]==SCHEMA
    assert r["real_FIT_target_support"]["real_FIT_physical_frames"]==325*300
    assert r["real_FIT_target_support"]["horizon_support"]["256"][
        "real_observed_state_action_next_state_pairs"]==325*44
    assert r["training_models_fitted"]==0
    assert not r["full_DEV_TEST_physical_source_qualified"]
    assert r["source_train_dev_test_episode_ID_hashes"]["FIT"] != r["source_train_dev_test_episode_ID_hashes"]["TEST"]
    with pytest.raises(SourceFailure,match="incomplete"):
        build_full_fit_receipt({},parts,digests,{i:x for i,x in details.items() if i!=next(iter(details))})


def test_no_dev_test_episode_can_masquerade_as_fit_episode():
    parts=partition_metadata(metadata_only_rows(all_original_shape_fixture()))
    fit={i:{} for i,_ in parts["FIT"]}
    fit[next(iter(parts["TEST"]))[0]]={}
    with pytest.raises(SourceFailure,match="incomplete|HELDOUT_LEAKAGE"):
        build_full_fit_receipt({},parts,{},fit)


def test_source_path_refuses_out_of_range_or_noninteger_episode_indices():
    assert source_path(504).endswith("episode_000504.parquet")
    for invalid in (-1,505,False,"2",2.0):
        with pytest.raises(SourceFailure):
            source_path(invalid)
