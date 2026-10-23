"""ME076 no-fit source/causality engineering checks; artificial fixtures NOT scientific evidence."""
from pathlib import Path
import json
import torch
import pytest

from scripts.me076_train_system1_real import (
    classify_members, build_batch, rollout, valid_origins, MOTION_PHYSICAL_THRESHOLD)
from scripts.validate_research_parity import check_manifest
from tac_osm.variants.system1_native import System1FastModel


def test_frozen_original_real_source_registration_and_true_100k_model():
    m=json.loads(Path("research/mathematical-engineering/me076/FAIR_COMPARISON.json").read_text())
    assert check_manifest(m,"preflight")["status"]=="PASS_PREFIT_SINGLE_MODEL"
    assert m["real_world_source"]["sha256"]=="4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776"
    assert m["real_world_source"]["observed_real_outcomes"] is True
    assert len(m["models"])==5
    assert all(v["trainable_parameters"]==100476 for v in m["models"])
    assert m["source_policy"]["CAL_or_unopened_target_access"] is False


def test_original_episodes_are_disjoint_and_task_balanced_without_opening_any_parquet():
    plan=json.loads(Path("research/mathematical-engineering/me064/TASK_LEVEL_SPLIT.json").read_text())
    fit,report=classify_members(plan)
    assert len(fit)==120 and len(report)==40
    assert len(set((t,e) for t,e,_ in fit))==120
    assert len(set((t,e) for t,e,_ in report))==40
    assert not set((t,e) for t,e,_ in fit)&set((t,e) for t,e,_ in report)
    assert set(t for t,e,_ in fit)==set(t for t,e,_ in report)
    for t in plan["training_tasks"]:
        assert sum(k==t for k,e,n in fit)==24
        assert sum(k==t for k,e,n in report)==8
    assert classify_members(plan)==(fit,report)


def sample_row(intervene=False):
    n=30
    state=torch.zeros(n,14)
    action=torch.ones(n,14)*.25
    for t in range(1,n):
        state[t]=.02*t
    human=torch.zeros(n,dtype=torch.bool)
    if intervene:human[16]=True
    return {"state":state,"action":action,"human":human,"n":n}


def test_real_input_contract_is_strictly_pre_outcome_and_lag1_known_action():
    source={("source",1):sample_row()}
    identified={"state_scale":[1.]*14}
    pools=valid_origins(source,identified)
    assert pools[("source",1)][0]
    time=12
    states,commands,truth=build_batch(source,[("source",1,time)])
    assert states.shape==(1,8,14) and commands.shape==(1,8,14)
    torch.testing.assert_close(states[0,-1],source[("source",1)]["state"][time])
    torch.testing.assert_close(commands[0,-1],source[("source",1)]["action"][time-1])
    torch.testing.assert_close(truth[0],source[("source",1)]["state"][time+1])
    m=System1FastModel(dimension=14)
    prediction=rollout(m,states,commands)
    src=sample_row()
    src["state"][time+1]=9999
    altered,act2,_=build_batch({("source",1):src},[("source",1,time)])
    torch.testing.assert_close(states,altered,atol=0,rtol=0)
    torch.testing.assert_close(commands,act2,atol=0,rtol=0)
    with torch.no_grad():
        other=rollout(m,altered,act2)
    torch.testing.assert_close(prediction.mean_next,other.mean_next,atol=0,rtol=0)


def test_human_takeover_source_masks_prevent_training_through_human_authority():
    source={("source",1):sample_row(intervene=True)}
    pools=valid_origins(source,{"state_scale":[1.]*14})
    eligible=pools[("source",1)][0]
    assert all(not (t-8<=16<=t+1) for t in eligible)
    assert any(t+1<16 for t in eligible)
    assert any(t-8>16 for t in eligible)


def test_no_new_results_or_training_is_claimed_by_the_fixture():
    file=Path("research/mathematical-engineering/me076/FAIR_COMPARISON.json")
    m=json.loads(file.read_text())
    assert m["source_policy"]["source_is_previously_exposed"] is True
    assert "DEV" in m["measurable_promotion_rule"]
    assert MOTION_PHYSICAL_THRESHOLD==0.025
