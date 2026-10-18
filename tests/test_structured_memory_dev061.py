"""Memory-isolation invariants; no superiority assertion from these unit tests."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from tac_osm.development.structured_memory import create,configurations
from tac_osm.development.learned_agent_world import event

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('dev061',ROOT/'scripts/run_structured_memory_dev061.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


@pytest.fixture(scope='module')
def settings():
    torch.set_num_threads(1)
    return configurations()


@pytest.mark.parametrize('arm',['generic','entity','entity_aging','entity_retention','entity_aging_retention','structured_gru','public_mlp'])
def test_cached_and_incremental_states_match(settings,arm):
    model=create(arm,settings[arm],81).eval()
    if arm!='public_mlp':
        runner.pretrain_admission(model,81,40)
        if model.retention:
            model.retention_ready.fill_(True)
            # Exercise selective writes, including a closed branch.
            with torch.no_grad():model.keep[-1].bias.fill_(-.05)
    ds=runner.raw_data(7161101,12)
    cached=runner.prepare(model,ds)
    with torch.no_grad():
        expected=model.training_state(cached);actual=model.initial(12)
        for t in range(ds['events'].shape[1]):actual,_=model.observe(actual,ds['events'][:,t],ds['present'][:,t])
    torch.testing.assert_close(actual[0],expected[0],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(actual[1],expected[1],atol=0,rtol=0)


def test_entity_update_cannot_corrupt_another_current_state(settings):
    model=create('entity',settings['entity']).eval()
    state=model.initial(1)
    with torch.no_grad():
        state,_=model.observe(state,torch.from_numpy(event(0,2,None,1,0))[None],force_write=True)
        before=state[0].clone()
        state,_=model.observe(state,torch.from_numpy(event(2,3,None,0,1))[None],force_write=True)
    torch.testing.assert_close(state[0][:,2],before[:,2],atol=0,rtol=0)
    assert not torch.equal(state[0][:,3],before[:,3])


def test_no_private_target_enters_memory_and_retention_gradients(settings):
    model=create('entity_retention',settings['entity_retention'],17)
    runner.pretrain_admission(model,17,40);model.retention_ready.fill_(True)
    raw=runner.raw_data(7161102,12);ds=runner.prepare(model,raw)
    changed=dict(ds,current=1-ds['current'],policy=ds['policy'].roll(1,-1))
    a=model.training_state(ds);b=model.training_state(changed)
    torch.testing.assert_close(a[0],b[0],atol=0,rtol=0)
    loss,_=runner.loss(model,ds,torch.arange(12),counterfactual=True)
    loss.backward()
    assert model.keep[-1].weight.grad.abs().sum()>0
    assert model.local.weight_ih_l0.grad.abs().sum()>0
    assert model.policy[-1].weight.grad.abs().sum()>0


def test_active_parameter_match_and_retrained_no_casm(settings):
    counts={a:sum(p.numel() for p in create(a,c).parameters()) for a,c in settings.items()}
    assert all(abs(n/counts['generic']-1)<=.05 for n in counts.values())
    assert not hasattr(create('structured_gru',settings['structured_gru']),'operator')


def test_public_feature_mlp_action_is_trainable_not_scripted(settings):
    model=create('public_mlp',settings['public_mlp'],17)
    ds=runner.prepare(model,runner.raw_data(7161103,12))
    value,_=runner.loss(model,ds,torch.arange(12));value.backward()
    assert model.action.weight.grad.abs().sum()>0
    assert model.network[0].weight.grad.abs().sum()>0


def test_frozen_wrapper_learned_path_is_neutral():
    from tac_osm.development.learned_agent import LearnedAgent,AgentConfig
    from tac_osm.development.frozen_read_diagnostic import FrozenReadDiagnostic
    config=AgentConfig(dimension=12);original=LearnedAgent(config,17).eval()
    wrapped=FrozenReadDiagnostic(config,17).eval();wrapped.load_state_dict(original.state_dict(),strict=True)
    a=original.initial(2);b=wrapped.initial(2);world=runner.ResearchWorld(7161104,0,4,32)
    with torch.no_grad():
        for e in world.events:
            e=torch.from_numpy(e)[None].expand(2,-1);a,_=original.observe(a,e);b,_=wrapped.observe(b,e)
        args=[torch.from_numpy(v)[None].expand(2,*v.shape) for v in world.candidates()[:4]]
        x=original(a,*args);y=wrapped(b,*args)
    torch.testing.assert_close(x['logits'],y['logits'],atol=0,rtol=0)
    torch.testing.assert_close(a[0],b[0][:,:,:12],atol=0,rtol=0)


def test_oracle_read_changes_freshness_only_between_forced_modes():
    from tac_osm.development.learned_agent import AgentConfig
    from tac_osm.development.frozen_read_diagnostic import FrozenReadDiagnostic
    model=FrozenReadDiagnostic(AgentConfig(dimension=12),17,'latest').eval();state=model.initial(1)
    events=[event(0,2,None,1,0),event(2,2,None,0,3),event(1,2,4,0,2)]
    # Construct a retained public bank directly; no hidden fault labels supplied.
    for i,e in enumerate(events):state[0][0,i,:12]=model.encoder(torch.from_numpy(e));state[0][0,i,12:]=torch.from_numpy(e);state[1][0,i]=1
    descriptors=torch.zeros(1,1,11);descriptors[0,0,2]=1;descriptors[0,0,5]=1
    model.descriptors=descriptors;action=model.action_encoder(descriptors)
    latest=model.read(state,action)[1].argmax(-1);model.mode='oldest';oldest=model.read(state,action)[1].argmax(-1)
    assert latest.tolist()==[[[1,2]]] and oldest.tolist()==[[[0,2]]]


def test_stop_metrics_keep_early_late_and_censored_cases_separate():
    from summarize_memory_dev061 import metrics
    def action(kind,bit):return {'kind':kind,'target':0,'before':[bit]}
    rows=[{'actions':[action('inspect',0),action('stop',0)],'success':True,'stopped':True,'final_faults':0},
          {'actions':[action('stop',1)],'success':False,'stopped':True,'final_faults':1},
          {'actions':[action('inspect',0)],'success':False,'stopped':False,'final_faults':0},
          {'actions':[action('repair',1)],'success':False,'stopped':False,'final_faults':0}]
    for r in rows:r.update(cost=0.,write_cost=0.)
    m=metrics([{'rows':rows}])
    assert m['stop_confusion']=={'tp':1,'fp':1,'fn':2,'tn':1}
    assert m['stop_precision']==.5 and m['stop_recall_per_healthy_decision']==1/3
    assert m['early_stop_worlds']==1 and m['missing_stop_worlds']==2
    assert m['lag_observed']==1 and m['lag_censored']==2
    assert m['lag_censored_lower_bounds']==[1,0]
    assert m['healthy_to_stop_lag_mean_observed']==1


def test_primary_paired_seed_test_keeps_ties_and_multiplicity():
    from summarize_memory_dev061 import signflip,holm
    assert signflip([0]*10)==1
    assert signflip([1]*8+[0]*2)==2/256
    assert holm({'a':.01,'b':.02,'c':.9,'d':1})=={'a':.04,'b':.06,'c':1.,'d':1.}


def test_inference_only_metadata_optimization_preserves_outputs_and_old_state(settings):
    from tac_osm.development.inference_memory import InferenceMemoryAgent
    from torch import nn
    old=create('generic',settings['generic'],17).eval();new=InferenceMemoryAgent(settings['generic'],'generic',17).eval()
    # A controlled public gate exercises both physical skip and write paths.
    for m in (old,new):
        m.admission=nn.Linear(22,1)
        with torch.no_grad():m.admission.weight.zero_();m.admission.weight[0,0]=8;m.admission.bias.fill_(-4)
    new.load_state_dict(old.state_dict());a=old.initial(2);b=new.initial(2)
    with torch.no_grad():
        for i in range(8):
            e=torch.from_numpy(event(0 if i%3==0 else 3,2,None,i%2,i))[None].expand(2,-1)
            original=b;snapshot=tuple(v.clone() for v in b);a,ta=old.observe(a,e);b,tb=new.observe(b,e)
            assert all(torch.equal(x,y) for x,y in zip(a,b))
            assert all(torch.equal(x,y) for x,y in zip(original,snapshot))
            if not ta['writes']:assert b is original
            assert torch.equal(ta['gate'],tb['gate'])
        w=runner.ResearchWorld(7161104,0,4,32);args=[torch.from_numpy(v)[None].expand(2,*v.shape) for v in w.candidates()[:4]]
        torch.testing.assert_close(old(a,*args)['logits'],new(b,*args)['logits'],rtol=0,atol=0)


def test_read_visibility_counts_dropped_latest_evidence_as_stale():
    from memory_metrics_dev061 import visibility
    records=torch.from_numpy(__import__('numpy').stack([event(0,2,None,1,0),event(2,2,None,0,3)]))[None]
    raw=torch.zeros(1,3,22);raw[0,0]=records[0,0];occupied=torch.tensor([[1.,0.,0.]])
    query=torch.zeros(1,1,8);query[0,0,2]=1
    r=visibility(raw,occupied,records,torch.ones(1,2,dtype=torch.bool),query,torch.ones(1,1,dtype=torch.bool),torch.tensor([[0]]))[0]
    assert r=={'queries':1,'stale':1,'latest_hits':0,'wrong':0,'latest_available':0}
    raw[0,1]=records[0,1];raw[0,2]=records[0,1];occupied[:]=1
    r=visibility(raw,occupied,records,torch.ones(1,2,dtype=torch.bool),query,torch.ones(1,1,dtype=torch.bool),torch.tensor([[2]]))[0]
    assert r=={'queries':1,'stale':0,'latest_hits':1,'wrong':0,'latest_available':1}


def test_public_feature_control_gets_equivalent_allocation_optimization(settings):
    from tac_osm.development.inference_public_features import InferencePublicFeatureMLP
    old=create('public_mlp',settings['public_mlp'],17).eval();new=InferencePublicFeatureMLP(settings['public_mlp']['width'],17).eval();new.load_state_dict(old.state_dict())
    a=old.initial(1);b=new.initial(1)
    with torch.no_grad():
        for i in range(8):
            e=torch.from_numpy(event(3 if i%2 else 0,2,None,i%2,i))[None];before=b
            a,_=old.observe(a,e);b,t=new.observe(b,e)
            assert all(torch.equal(x,y) for x,y in zip(a,b))
            if not t['writes']:assert b is before
        w=runner.ResearchWorld(7161104,0,4,32);args=[torch.from_numpy(v)[None] for v in w.candidates()[:4]]
        torch.testing.assert_close(old(a,*args)['logits'],new(b,*args)['logits'],atol=0,rtol=0)


def test_exact_retention_preserves_timestamp_keys_and_gate_gradients(settings):
    from tac_osm.development.exact_retention import ExactRetentionAgent
    p=torch.tensor(.55,requires_grad=True);gate=torch.ones_like(p)+(p-p.detach());assert gate.item()==1.;gate.backward();assert p.grad.item()==1.
    assert (torch.ones_like(p)+p-p.detach()).item()!=1.
    model=ExactRetentionAgent(settings['entity_retention'],'entity_retention',17).eval()
    runner.pretrain_admission(model,17,40);model.retention_ready.fill_(True)
    with torch.no_grad():
        for value in model.keep.parameters():value.zero_()
        model.keep[-1].bias.fill_(float(torch.logit(torch.tensor(.55))))
        ds=runner.raw_data(7161105,12);cached=runner.prepare(model,ds);a=model.training_state(cached);b=model.initial(12)
        for t in range(ds['events'].shape[1]):b,_=model.observe(b,ds['events'][:,t],ds['present'][:,t])
        torch.testing.assert_close(a[0],b[0],atol=1e-6,rtol=1e-5);torch.testing.assert_close(a[1],b[1],atol=0,rtol=0)
        for i in range(12):
            for slot in (a[1][i,16:32]>0).nonzero().flatten()+16:
                assert (cached['signals'][i,:,-1]==a[0][i,slot,-1]).any()
    model.train();value,_=runner.loss(model,cached,torch.arange(12),counterfactual=True);value.backward()
    assert model.keep[-1].bias.grad.abs().sum()>0
