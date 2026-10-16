import math
import pytest
import torch
from tac_osm.development.gradient_diagnostics import gradient_health
from tac_osm.development.integration_test6 import MaintenanceSession,PublicRequest,Test6Controller,public_tokens
from tac_osm.development.schema_maintenance import make_pair


def test_finite_gradient_norm_overflow_is_not_mislabeled_nan():
    p=torch.nn.Parameter(torch.ones(2));p.grad=torch.tensor([1e20,1e20]);before=p.grad.clone()
    with pytest.raises(RuntimeError):torch.nn.utils.clip_grad_norm_([p],1.,error_if_nonfinite=True)
    health=gradient_health([('fixture',p)])
    assert health['classification']=='finite_gradients_float32_norm_overflow'
    assert health['nonfinite_elements']==0 and math.isfinite(health['total_norm_float64'])
    assert torch.equal(p.grad,before)


def test_nonfinite_elements_are_a_separate_failure():
    p=torch.nn.Parameter(torch.ones(2));p.grad=torch.tensor([float('nan'),float('inf')])
    report=gradient_health([('fixture',p)])
    assert report['classification']=='nonfinite_gradient_elements'
    assert report['nonfinite_elements']==2


def test_public_encoder_cannot_accept_an_episode():
    case=make_pair(1,32,32,'retention','at',domain='dev032-test')[0]
    with pytest.raises(TypeError):public_tokens(case,PublicRequest(0,32))


def test_duplicate_service_is_charged_and_actions_are_bounded():
    case=make_pair(1,32,32,'retention','at',domain='dev032-test')[0]
    env=MaintenanceSession(case,PublicRequest(case.target_entity,case.horizon))
    env.act(1);env.act(1);score=env.score()
    assert score['services']==2 and score['total_cost']==2
    with pytest.raises(ValueError):env.act(0)


def test_repair_executes_and_rejected_commit_is_identity():
    torch.set_num_threads(1);controller=Test6Controller()
    case=make_pair(1,32,32,'pending_cancel','at',domain='dev032-test')[0]
    env=MaintenanceSession(case,PublicRequest(case.target_entity,case.horizon))
    result=controller.rollout(env,force_gate=0.)
    assert result['repair_executed'] and len(env.receipts)==2
    assert result['rejected_commits']==2 and result['forced_gate_fixture']
    assert result['capability_validated'] is False


def test_repair_off_never_executes_a_second_action():
    torch.set_num_threads(1);controller=Test6Controller()
    case=make_pair(1,32,32,'pending_cancel','at',domain='dev032-test')[0]
    env=MaintenanceSession(case,PublicRequest(case.target_entity,case.horizon))
    result=controller.rollout(env,arm='repair_off',force_gate=0.)
    assert not result['repair_executed'] and len(env.receipts)==1
