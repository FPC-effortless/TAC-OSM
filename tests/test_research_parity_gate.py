"""Safety tests for preregistered FAIR architecture comparisons.

All generated numbers below are strictly software fixtures, NOT actual model
fits or real-world evidence. Any failing condition disqualifies science claims.
"""
from copy import deepcopy
import pytest
from scripts.validate_research_parity import (
    check_manifest, ParityError, check_model_parameter_target,
    requires_100k_policy
)

def source_hash(ch):return ch*64

def fixture():
    digests={k:source_hash(str(i%10)) for i,k in enumerate((
        "dataset_sha256","train_split_sha256","dev_split_sha256",
        "test_split_sha256","training_sample_order_sha256",
        "feature_contract_sha256","target_contract_sha256",
        "optimizer_schedule_sha256","tuning_policy_sha256",
        "selection_policy_sha256","evaluation_contract_sha256"))}
    digests["dataset_sha256"]=source_hash("a")
    arms=[]
    for seed in (101,102):
        for name,parameters in (("TAC native",100000),("GRU control",100400)):
            b={"steps":500,"examples_seen":32000,"tokens_seen":2048000,
               "batch_size":64,"history_events":64,"planned_training_flops":1_000_000_000,
               "max_training_flops":1_100_000_000,"max_optimizer_cpu_seconds":70,
               "tuning_trials":0,"tuning_flops":0,
               "prefit_profile_training_flops":1_000_000_000,
               "prefit_profile_optimizer_cpu_seconds":50.0,
               "prefit_profile_method":"torch.profiler comparable real-input dry-run"}
            arms.append({"architecture":name,"seed":seed,
              "trainable_parameters":parameters,"deployed_parameters":parameters,
              "nontrainable_state_bytes":8192,"hardware_profile":"1-core CPU AVX2 pytorch2 pinned",
              **digests,"budget":b,
              "measured":{"trainable_parameters":parameters,"deployed_parameters":parameters,
                  "actual_steps":500,"actual_examples":32000,"actual_tokens":2048000,
                  "training_flops":1_000_000_000,"optimizer_cpu_seconds":50.0,
                  "peak_memory_bytes":2_000_000,"flops_profiler":"torch.profiler CPU traced all kernels",
                  "checkpoint_sha256":source_hash("e"),
                  "actual_sample_order_sha256":digests["training_sample_order_sha256"],
                  "evaluation_split_sha256":digests["test_split_sha256"]}})
    return {"schema_version":"tac-osm.parity.v1","study_id":"ME999_SOFTWARE_FIXTURE",
      "study_type":"architecture_comparison","registered_before_first_fit":True,
      "real_world_source":{"name":"Declared original publisher real source","sha256":source_hash("a"),
                           "license":"Original observed license","observed_real_outcomes":True},
      "literature_gap":{"unresolved_TAConly_question":"Test distinctive TAC decision memory on real events",
        "what_is_not_being_rediscovered":"Do not re-test generic GRU temporal context advantages",
        "established_work":[{"paper":"Established recurrent state learning","url":"https://example.org/p1",
          "established_result":"Already shown that recurrent models can process histories"},
          {"paper":"Established world model dynamics","url":"https://example.org/p2",
          "established_result":"Already shown that action-conditioned dynamics are feasible"}]},
       "arms":arms}


def test_preflight_software_manifest_passes_only_when_all_dimensions_matched():
    m=fixture()
    assert check_manifest(m,"preflight")["status"]=="PASS"
    assert not check_manifest(m,"preflight")["eligible_for_architecture_superiority"]

def test_final_requires_real_measurement_fields_and_replay_receipts():
    m=fixture()
    assert check_manifest(m,"final")["eligible_for_architecture_superiority"]
    del m["arms"][1]["measured"]["checkpoint_sha256"]
    with pytest.raises(ParityError,match="checkpoint"):check_manifest(m,"final")

def test_preflight_rejects_parameter_mismatch_not_merely_warns():
    m=fixture()
    m["arms"][1]["trainable_parameters"]=102000
    m["arms"][1]["deployed_parameters"]=102000
    with pytest.raises(ParityError,match="parameters mismatch"):check_manifest(m)

def test_preflight_rejects_source_or_actual_exposure_mismatch():
    m=fixture()
    m["arms"][0]["training_sample_order_sha256"]=source_hash("c")
    with pytest.raises(ParityError,match="training_sample_order_sha256"):check_manifest(m)
    m=fixture()
    m["arms"][0]["budget"]["examples_seen"]=31000
    with pytest.raises(ParityError,match="examples_seen"):check_manifest(m)

def test_preflight_rejects_planned_compute_inequality():
    m=fixture()
    m["arms"][1]["budget"]["planned_training_flops"]=1_120_000_000
    m["arms"][1]["budget"]["max_training_flops"]=1_120_000_000
    with pytest.raises(ParityError,match="max_training_flops"):check_manifest(m)

def test_postfit_rejects_unmatched_training_compute_not_cpu_speed():
    m=fixture()
    m["arms"][1]["measured"]["training_flops"]=1_100_000_000
    with pytest.raises(ParityError,match="actual FLOPs"):check_manifest(m,"final")
    m=fixture()
    # Runtime is efficiency reporting, NOT a mandate that unlike models
    # take the same number of CPU seconds. This previously failed incorrectly.
    m["arms"][1]["measured"]["optimizer_cpu_seconds"]=580.0
    assert check_manifest(m,"final")["status"]=="PASS"

def test_training_sample_order_evaluation_ids_and_tuning_privileges_are_hard_gates():
    m=fixture()
    m["arms"][1]["tuning_policy_sha256"]=source_hash("b")
    with pytest.raises(ParityError,match="tuning_policy_sha256"):check_manifest(m)
    m=fixture()
    m["arms"][1]["measured"]["actual_sample_order_sha256"]=source_hash("b")
    with pytest.raises(ParityError,match="sample order"):check_manifest(m,"final")
    m=fixture()
    m["arms"][1]["measured"]["evaluation_split_sha256"]=source_hash("b")
    with pytest.raises(ParityError,match="test IDs"):check_manifest(m,"final")

def test_literature_gap_or_required_seed_cannot_be_omitted():
    m=fixture()
    m["literature_gap"]["established_work"]=[]
    with pytest.raises(ParityError,match="prior research"):check_manifest(m)
    m=fixture()
    m["arms"]=[a for a in m["arms"] if a["seed"]==101]
    with pytest.raises(ParityError,match="need >=2 architecture arms x >=2 seeds|two controlled independent seeds"):
        check_manifest(m)

def test_diagnostic_cannot_smuggle_architecture_superiority_label():
    m={"schema_version":"tac-osm.parity.v1","study_id":"REAL_DIAGNOSTIC",
       "study_type":"noncomparative_diagnostic","scientific_purpose":"Investigate numerical replay",
       "no_architecture_superiority_claim":True,
       "no_new_model_fits":True}
    assert check_manifest(m)["status"]=="PASS_DIAGNOSTIC_ONLY"
    m["no_architecture_superiority_claim"]=False
    with pytest.raises(ParityError,match="forbid"):check_manifest(m)


def test_prefit_cpu_duration_mismatch_is_allowed_when_parameters_data_budget_match():
    m=fixture()
    m["arms"][1]["budget"]["prefit_profile_optimizer_cpu_seconds"]=700.0
    m["arms"][1]["budget"]["max_optimizer_cpu_seconds"]=700
    assert check_manifest(m,"preflight")["status"]=="PASS"


def test_prefit_blocks_training_when_measured_flops_pilot_differs():
    m=fixture()
    m["arms"][1]["budget"]["prefit_profile_training_flops"]=1_120_000_000
    with pytest.raises(ParityError,match="PROFILER prefit FLOPs"):
        check_manifest(m,"preflight")

def test_identical_training_budget_can_pass_without_a_cpu_time_limit():
    m=fixture()
    for arm in m["arms"]:
        arm["budget"].pop("max_optimizer_cpu_seconds")
        arm["budget"].pop("prefit_profile_optimizer_cpu_seconds")
    assert check_manifest(m,"preflight")["status"]=="PASS"
    assert check_manifest(m,"final")["status"]=="PASS"


def test_new_studies_hard_gate_absolute_parameter_target_even_if_parity_matches():
    m=fixture()
    assert requires_100k_policy("ME073_RESEARCH_MODEL")
    for row in m["arms"]:
        row["trainable_parameters"]=38000
        row["deployed_parameters"]=38000
    with pytest.raises(ParityError,match="100k target"):
        check_manifest(m,"preflight")
    for row in m["arms"]:
        row["trainable_parameters"]=123377
        row["deployed_parameters"]=123377
    with pytest.raises(ParityError,match="100k target"):
        check_manifest(m,"preflight")


def test_new_model_size_limits_are_inclusive_and_bound_frozen_learned_capacity():
    for amount in (95000,100000,105000):
        check_model_parameter_target(amount,amount)
    for amount in (94999,105001):
        with pytest.raises(ParityError,match="100k target"):
            check_model_parameter_target(amount)
    with pytest.raises(ParityError,match="100k target"):
        check_model_parameter_target(100000,120000)


def test_historical_original_study_is_not_retroactively_disqualified():
    m=fixture()
    m["study_id"]="ME066_FOUR_ARM_GENUINE_ORIGINAL_ROBOT_FIT_EQUAL_BUDGET"
    for row in m["arms"]:
        row["trainable_parameters"]=37177
        row["deployed_parameters"]=37177
    # Equal counts remain mandatory; absolute policy only governs new fits.
    assert not requires_100k_policy(m["study_id"])
    assert check_manifest(m,"preflight")["status"]=="PASS"


def test_new_single_model_uses_same_absolute_100k_rule_and_final_real_count():
    m={
        "schema_version":"tac-osm.parity.v1",
        "study_id":"ME073_SINGLE_MODEL",
        "study_type":"single_model_training",
        "registered_before_first_fit":True,
        "real_world_source":{"name":"Authentic real source","sha256":source_hash("a"),
                             "observed_real_outcomes":True},
        "models":[{"seed":101,"trainable_parameters":100000,
                   "deployed_parameters":100000}]
    }
    assert check_manifest(m,"preflight")["status"]=="PASS_PREFIT_SINGLE_MODEL"
    with pytest.raises(ParityError,match="actual measurement"):
        check_manifest(m,"final")
    m["models"][0]["measured"]={
        "trainable_parameters":100000,"deployed_parameters":100000,
        "checkpoint_sha256":source_hash("b")}
    assert check_manifest(m,"final")["status"]=="PASS_SINGLE_MODEL"
    m["models"][0]["measured"]["trainable_parameters"]=105001
    with pytest.raises(ParityError,match="100k target"):
        check_manifest(m,"final")
    m["models"][0].pop("measured")
    m["models"][0]["trainable_parameters"]=47000
    m["models"][0]["deployed_parameters"]=47000
    with pytest.raises(ParityError,match="100k target"):
        check_manifest(m,"preflight")


def test_new_noncomparative_diagnostic_cannot_hide_new_model_fits():
    m={
      "schema_version":"tac-osm.parity.v1",
      "study_id":"ME073_SOURCE_ONLY_DIAGNOSTIC",
      "study_type":"noncomparative_diagnostic",
      "scientific_purpose":"Read actual original event timing with zero model updates",
      "no_architecture_superiority_claim":True,
    }
    with pytest.raises(ParityError,match="no-fit"):
        check_manifest(m)
    m["no_new_model_fits"]=True
    assert check_manifest(m)["status"]=="PASS_DIAGNOSTIC_ONLY"


def test_new_study_above_target_still_invalid_when_arms_match_to_1_percent():
    m=fixture()
    for row in m["arms"]:
        row["trainable_parameters"]=104900
        row["deployed_parameters"]=104900
    assert check_manifest(m)["status"]=="PASS"
    # Each new model could be near 110k and matched to its comparator,
    # but the prospective 100k absolute model-size gate rejects all.
    for row in m["arms"]:
        row["trainable_parameters"]=109500
        row["deployed_parameters"]=109500
    with pytest.raises(ParityError,match="100k target"):
        check_manifest(m)


# The prospective 100k limit is independent of the 1% between-arm parity check.
def test_new_study_ids_and_unknown_studies_cannot_escape_100k_training_gate():
    for name in ("ME073_NEXT_PHYSICAL_MODEL", "DEV062_NEW_MODEL", "NEW_RECURRENT_WORLD_MODEL"):
        assert requires_100k_policy(name)
    assert not requires_100k_policy("ME072_HISTORICAL_ALREADY_FIT")
    assert not requires_100k_policy("DEV061_ORIGINAL_HISTORICAL")
    # Even two identically undersized models cannot qualify in a new study.
    m=fixture()
    m["study_id"]="ME073_NEXT_PHYSICAL_MODEL"
    for arm in m["arms"]:
        arm["trainable_parameters"]=43000
        arm["deployed_parameters"]=43000
    with pytest.raises(ParityError,match="100k target"):
        check_manifest(m,"preflight")
