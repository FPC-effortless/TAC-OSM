import copy
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from tac_osm.development.reliability_controllers import configurations,create,DenseController
from tac_osm.development.learned_agent_world import World,teacher
SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from run_reliability_controls_dev059 import data,retained_bank,task_loss,raw_kind_pretraining,evaluate
from run_fully_learned_dev058 import writer_fit
from audit_evidence_gaps_dev059 import public_reference,wilson


@pytest.fixture(scope='module')
def specs():
    torch.set_num_threads(1)
    return configurations()


def test_active_parameters_matched_and_casm_actually_removed(specs):
    for arm,spec in specs.items():
        model=create(arm,spec,8591001)
        assert abs(sum(p.numel() for p in model.parameters())/133580-1)<=.05
        if arm=='no_casm_stop':
            assert not any(name.startswith(('operator','graph_','halt')) for name,_ in model.named_parameters())


def test_raw_control_training_and_incremental_state_agree(specs):
    ds=data(8591002,6)
    for arm in ('gru_stop','attention_stop'):
        model=create(arm,specs[arm],8591003).eval()
        assert isinstance(model,DenseController)
        with torch.no_grad():
            training=model.training_state(ds)
            stream=model.initial(6)
            for t in range(ds['events'].shape[1]):stream,_=model.observe(stream,ds['events'][:,t],ds['present'][:,t])
            expected=model(training,ds['descriptors'],ds['valid'],ds['queries'],ds['query_valid'])
            actual=model(stream,ds['descriptors'],ds['valid'],ds['queries'],ds['query_valid'])
            torch.testing.assert_close(expected['logits'],actual['logits'],rtol=2e-5,atol=2e-5)


def test_all_arms_receive_common_task_targets_and_gradients(specs):
    raw=data(8591004,8)
    for arm,spec in specs.items():
        model=create(arm,spec,8591005)
        if isinstance(model,DenseController):
            assert raw_kind_pretraining(model,8591005,2)['updates']==2
            ds=raw
        else:
            writer_fit(model,8591005,20);ds=retained_bank(model,raw)
        loss,metrics=task_loss(model,ds,torch.arange(8),4)
        loss.backward()
        assert torch.isfinite(loss)
        assert model.policy[0].weight.grad.abs().sum()>0
        assert model.transition[0].weight.grad.abs().sum()>0
        assert 'healthy_stop_accuracy' in metrics


def test_stop_weight_changes_training_loss_without_runtime_stop_rule(specs):
    ds=data(8591006,12);model=create('gru_stop',specs['gru_stop'],8591007).eval()
    ids=torch.arange(12)
    l1,m1=task_loss(model,ds,ids,1);l4,m4=task_loss(model,ds,ids,4)
    assert ds['completion'].sum()>0 and l4>l1
    assert m1['policy_ce']==m4['policy_ce']
    state=model.training_state(ds)
    one=model(state,ds['descriptors'],ds['valid'],ds['queries'],ds['query_valid'])['logits']
    two=model(state,ds['descriptors'],ds['valid'],ds['queries'],ds['query_valid'])['logits']
    torch.testing.assert_close(one,two,rtol=0,atol=0)


def test_teacher_equals_public_reference_and_physically_correct_stop():
    for i in range(12):
        w=World(8591008,i,6,96)
        for _ in range(16):
            desc,valid,_,_,commands=w.candidates()
            choice=public_reference(w.events,desc,valid)
            assert choice==int(teacher(w.bits,w.parents,commands).argmax())
            w.act(commands[choice])
            if w.stopped:
                assert not w.bits.any();break


def test_privileged_teacher_actions_can_be_unidentifiable_from_public_history():
    a=World(8591009,0,2,32);a.parents[:]=-1
    a.events=[e for e in a.events if e[3]>.5]
    a.bits[:]=[0,1];b=copy.deepcopy(a);b.bits[:]=[1,0]
    ad,av,aq,aqv,commands=a.candidates();bd,bv,bq,bqv,_=b.candidates()
    for x,y in zip((ad,av,aq,aqv),(bd,bv,bq,bqv)):np.testing.assert_array_equal(x,y)
    assert np.array_equal(a.events,b.events)
    assert int(teacher(a.bits,a.parents,commands).argmax())!=int(teacher(b.bits,b.parents,commands).argmax())
    with pytest.raises(ValueError,match='public initial'):public_reference(a.events,ad,av)


def test_wilson_conditional_interval_and_evaluation_physical_metric(specs):
    lo,hi=wilson(576,576)
    assert .993<lo<.994 and hi==pytest.approx(1)
    model=create('gru_stop',specs['gru_stop'],8591010).eval()
    cell=evaluate(model,8591011,2,32,2)
    assert cell['physical_repair']>=cell['completion']
    assert all('initial_events' not in row for row in cell['rows'])
    assert all(len(row['public_initial_sha256'])==64 for row in cell['rows'])


def test_replay_tolerance_applies_only_to_float_telemetry():
    from audit_reliability_controls_dev059 import compare
    reference={'rows':[{'actions':[{'kind':'stop'}],'cost':1.,'completion_probabilities':[.5],'stop_probabilities':[.6]}]}
    actual=copy.deepcopy(reference);actual['rows'][0]['completion_probabilities'][0]+=1e-7
    compare(actual,reference)
    drift={'values':0,'max_absolute_difference':0.,'outside_initial_tolerance':0}
    actual['rows'][0]['completion_probabilities'][0]=.50001
    compare(actual,reference,drift)
    assert drift['outside_initial_tolerance']==1 and drift['max_absolute_difference']>0
    actual['rows'][0]['completion_probabilities'][0]=.51
    with pytest.raises(AssertionError):compare(actual,reference)
    actual['rows'][0]['completion_probabilities'][0]=.5
    actual['rows'][0]['actions'][0]['kind']='repair'
    with pytest.raises(AssertionError,match='discrete'):compare(actual,reference)


def test_all_control_families_export_without_research_imports(tmp_path,specs):
    from audit_reliability_controls_dev059 import standalone
    for arm,spec in specs.items():
        model=create(arm,spec,8591016).eval();checkpoint=tmp_path/f'{arm}.pt'
        torch.save(model.state_dict(),checkpoint)
        assert standalone(model,arm,spec,checkpoint)>=3


def test_repair_routing_labels_use_public_records_not_private_task_targets(specs):
    from repair_learned_checkpoint_dev059 import public_routing_targets,repair_loss
    raw=data(8591017,8);model=create('full_stop',specs['full_stop'],8591018)
    writer_fit(model,8591018,20);ds=retained_bank(model,raw)
    first=public_routing_targets(ds,32)
    changed=dict(ds,policy=torch.zeros_like(ds['policy']),one=torch.ones_like(ds['one']))
    second=public_routing_targets(changed,32)
    assert torch.equal(first['attention'],second['attention']) and torch.equal(first['graph'],second['graph'])
    loss,metrics=repair_loss(model,first,torch.arange(8));loss.backward()
    assert torch.isfinite(loss) and metrics['extra_stop_ce']>=0
    assert model.read_query.weight.grad.abs().sum()>0
