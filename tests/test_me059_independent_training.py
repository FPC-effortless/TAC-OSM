"""ME059 numerical/software contract tests. No synthetic capability score claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.me059_train_independent_robot import allow_fit, score_episode
from scripts import me058_profile_original_fit as base


def fake_original(n=850):
    rng=np.random.default_rng(12)
    x=np.cumsum(rng.normal(size=(n,14)).astype(np.float32)*.001,axis=0)
    a=x+rng.normal(size=(n,14)).astype(np.float32)*.0004
    return {"state":x,"action":a,"human":np.zeros(n,dtype=np.bool_)}


def test_me059_must_not_fit_with_source_diagnostic_or_unverified_preflight():
    diagnostic={"study_type":"noncomparative_diagnostic",
                "training_authorized":False}
    with pytest.raises(RuntimeError,match="NO_TRAINING"):
        allow_fit(diagnostic)
    unfinished={"study_type":"architecture_comparison",
                "training_authorized":True,
                "schema_version":"tac-osm.parity.v3",
                "fit_source_manifest_sha256":base.SOURCE_MANIFEST_SHA256,
                "frozen_sample_orders":{str(s):"0"*64 for s in base.SEEDS},
                "expected_physical_prior_sha256":"0"*64,
                "optimizer_updates":base.UPDATES,
                "batch_size":base.BATCH_SIZE,
                "history_frames":base.HISTORY,
                "forecast_frames":base.HORIZON,
                "me058_source_backed_FLOP_preflight_pass":False}
    with pytest.raises(RuntimeError,match="NO_TRAINING"):
        allow_fit(unfinished)


def test_frozen_preregistered_contract_sha256_matches_raw_text():
    path=Path("research/mathematical-engineering/me059")
    manifest=json.loads((path/"FAIR_COMPARISON.json").read_text())
    contracts=json.loads((path/"CONTRACTS.json").read_text())
    mapping={"features":"feature_contract_sha256",
             "targets":"target_contract_sha256",
             "optimizer":"optimizer_schedule_sha256",
             "tuning":"tuning_policy_sha256",
             "selection":"selection_policy_sha256",
             "evaluation":"evaluation_contract_sha256"}
    for key,digest_name in mapping.items():
        expected=hashlib.sha256(contracts[key].encode()).hexdigest()
        assert {a[digest_name] for a in manifest["arms"]}=={expected}
    assert manifest["full_FLOP_certification_available"] is False
    assert manifest["cpu_time_matching_required"] is False


def test_factual_evaluator_censors_after_first_human_takeover():
    original=fake_original()
    original["human"][257:]=True  # one valid physical step after t=256
    prior=base.identify_fit_only({0:fake_original()})
    model=base.construct("gru_world",prior)
    result=score_episode("gru_world",model,original,prior)
    assert result["origins"]==1
    assert result["counts"][0]==14
    assert sum(result["counts"][1:])==0
    assert result["onset_positives"]==1
    assert set(result["errors"])=={"gru_world","persistence","servo_prior"}


def test_scoring_never_fabricates_missing_neural_rival_results():
    original=fake_original()
    prior=base.identify_fit_only({0:original})
    model=base.construct("native_tac",prior)
    result=score_episode("native_tac",model,original,prior)
    assert result["origins"]==3
    assert result["counts"][0]==42
    assert "gru_world" not in result["errors"]
