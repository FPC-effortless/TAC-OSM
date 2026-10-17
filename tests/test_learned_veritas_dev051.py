"""DEV-051 public-observation policy, teacher and safety evaluation contracts."""
import pytest
torch=pytest.importorskip("torch")
from tac_osm.development.learned_veritas_policy import (
    ACTION_ID,FEATURE_DIM,PublicBelief,PublicActionPolicy,
    public_teacher,choose_action,fit,rollout,public_cohort)


def test_public_features_fixed_without_hidden_identity_or_future():
    b=PublicBelief(1)
    assert len(b.features())==FEATURE_DIM==12
    assert all(type(x) is float for x in b.features())
    assert b.known_fault is None and b.attempts==0
    assert b.features()==PublicBelief(1).features()


def test_teacher_uses_only_inspections_and_accepted_receipts():
    state=PublicBelief(1)
    assert public_teacher(state)==ACTION_ID["inspect"]
    state.observe("inspect",{"fault":1,"accepted":True})
    assert public_teacher(state)==ACTION_ID["isolate"]
    state.observe("isolate",{"accepted":True})
    assert public_teacher(state)==ACTION_ID["service"]
    state.observe("service",{"accepted":True})
    assert state.known_fault is None and state.attempts==1 and not state.isolated
    assert public_teacher(state)==ACTION_ID["inspect"]
    state.observe("inspect",{"fault":0,"accepted":True})
    assert public_teacher(state)==ACTION_ID["done"]


def test_constraints_exclude_unisolated_service_but_not_early_unsafe_done():
    x=PublicBelief(0)
    assert x.legal()==[True,True,False,True]
    x.observe("isolate",{"accepted":True})
    assert x.legal()==[True,False,True,True]
    x.observe("service",{"accepted":True})
    assert x.legal()==[True,True,False,True]


def test_model_and_fallback_are_separated():
    m=PublicActionPolicy(7101)
    state=PublicBelief(1)
    assert choose_action(state,None)[0]==ACTION_ID["inspect"]
    action,used,confidence=choose_action(state,m,1.01)
    assert action==ACTION_ID["inspect"] and used
    assert 0<=confidence<=1
    assert choose_action(state,m,0.)[1] is False


def test_policy_head_trains_on_public_observation_targets():
    model=PublicActionPolicy(7101)
    state=PublicBelief(1)
    x=torch.tensor([state.features()],dtype=torch.float32)
    y=torch.tensor([ACTION_ID["inspect"]])
    before=torch.nn.functional.cross_entropy(model.logits(x),y)
    _,curve=fit(model,x,y,10)
    after=torch.nn.functional.cross_entropy(model.logits(x),y)
    assert after<before and curve[-1]["step"]==10


def test_public_veritas_rollout_matches_reference_without_privileged_input():
    pytest.importorskip("investigation_world.operational.runtime")
    from tac_osm.development.veritas_public_challenge import make_episode,run_episode
    episode=make_episode(51101,0)
    payload=episode.public_payload()
    assert "oracle" not in payload
    learned,features=rollout(episode,"public_teacher",collect=True)
    reference=run_episode(episode,"inspect_verify_retry")
    assert learned["failures"]==reference["failures"]==0
    assert learned["total_cost"]==reference["total_cost"]
    assert learned["blocked"]==0
    assert features and all(len(row["x"])==12 for row in features)
    assert learned["complete"]==1


def test_development_splits_do_not_share_world_ids():
    pytest.importorskip("investigation_world.operational.runtime")
    pools=[list(public_cohort([seed],2)) for seed in (51101,51201,52101,53101)]
    ids=[set(ep.public_payload()["episode_id"] for _,_,ep in pool) for pool in pools]
    assert all(ids[i].isdisjoint(ids[j]) for i in range(4) for j in range(i))


def test_replay_of_same_policy_has_same_native_physical_results():
    pytest.importorskip("investigation_world.operational.runtime")
    from tac_osm.development.veritas_public_challenge import make_episode
    m=PublicActionPolicy(7102)
    a,_=rollout(make_episode(51102,1),"learner",m)
    b,_=rollout(make_episode(51102,1),"learner",m)
    assert a==b


def test_rejected_isolation_receipt_never_claims_slot():
    state=PublicBelief(1)
    state.observe("isolate",{"accepted":False})
    assert not state.isolated
    assert state.attempts==0
    assert public_teacher(state)==ACTION_ID["inspect"]
