import itertools
import pytest
import torch
from tac_osm.development.optimization_program import (
    ARMS, LEARNING_RATES, CURRICULA, BUDGETS, EventTablePredictor,
    domain, history_at, select_recipes)
from tac_osm.development.schema_maintenance import make_pair, ROLES, DEADLINES


def encode(streams):
    # Independent public encoding; no evaluator labels in the model input.
    kinds=('inspect','change','scheduled','wait')
    features=[]
    for stream in streams:
        row=[]
        for o in stream:
            v=[0.]*16;v[o.entity]=1;v[8+kinds.index(o.kind)]=1
            v[12]=o.tick/2048;v[13]=max(o.due,0)/2048;v[14+max(0,o.value)]=1
            row.append(v)
        features.append(row)
    return torch.tensor(features),torch.tensor([[o.entity for o in s] for s in streams]),torch.tensor([[max(o.value,0) for o in s] for s in streams])


def test_partition_domains_and_fingerprints_are_disjoint():
    banks=[{c.fingerprint for c in make_pair(1,32,32,'retention','at',domain=domain(split))}
           for split in ('train','validation','test')]
    assert all(not a&b for a,b in itertools.combinations(banks,2))
    with pytest.raises(ValueError):domain('typo')


def test_curriculum_does_not_change_at_pilot_budget():
    assert history_at('128_then512',215)==128
    assert history_at('128_then512',719)==128
    assert history_at('128_then512',720)==512
    assert history_at('fixed512',0)==512


def test_table_matches_public_causal_sufficient_state():
    model=EventTablePredictor()
    for role,deadline in itertools.product(ROLES,DEADLINES):
        cases=make_pair(419,32,32,role,deadline,domain='dev029-integrity')
        state=model.encode(*encode([c.observations for c in cases]))
        slots=state[:,:40].reshape(-1,8,5)
        for b,c in enumerate(cases):
            assert tuple(int(x) for x in slots[b,:,0])==c.current_targets
            future=slots[b,:,0].clone()
            for e in range(8):
                bit,generation,due,value,scheduled=slots[b,e].tolist()
                if due>c.tick and due<=c.tick+c.horizon and generation==scheduled:
                    future[e]=value
            assert tuple(int(x) for x in future)==c.wait_targets
        assert state.shape==(2,41)


def test_table_head_is_learned_and_independent_of_hidden_world():
    cases=make_pair(7,32,32,'retention','at',domain='dev029-integrity')
    model=EventTablePredictor();state=model.encode(*encode([c.observations for c in cases]))
    q=torch.zeros(2,8,3,14);logits=model.decode(state,q)
    logits.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.head.parameters())
    assert model.state_scalars==41 and logits.shape==(2,8,3,2)


def records():
    return [dict(arm=a,learning_rate=lr,curriculum=c,seed=s,updates=b,
                 split='validation',smoke=False,primary_accuracy=.5,cross_entropy=.7)
            for a,lr,c,s,b in itertools.product(ARMS,LEARNING_RATES,CURRICULA,(311,419,557),BUDGETS)]


def test_selection_requires_complete_matrix_and_rejects_test():
    rows=records()
    with pytest.raises(ValueError):select_recipes(rows[:-1])
    rows[0]['split']='test'
    with pytest.raises(ValueError):select_recipes(rows)


def test_selection_uses_final_validation_only_and_stable_ties():
    rows=records()
    for r in rows:
        if r['updates']<2160:r['primary_accuracy']=1.
    chosen=select_recipes(rows)
    assert all(v==dict(learning_rate=.0005,curriculum='fixed512',updates=2160) for v in chosen.values())
    for r in rows:
        if r['arm']=='native' and r['learning_rate']==.002 and r['updates']==2160:
            r['primary_accuracy']=.8
    assert select_recipes(rows)['native']['learning_rate']==.002


def test_tuning_resume_preserves_optimizer_and_exact_final_parameters(tmp_path,monkeypatch):
    # Exercise actual updates and optimizer serialization on short public fixture
    # histories, never registered validation/test cases. Compare uninterrupted
    # training with a stopped-and-resumed trial, not expected parameter values.
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    script=Path(__file__).resolve().parents[1]/'scripts/run_optimization_study_dev029.py'
    spec=importlib.util.spec_from_file_location('dev029_study_resume_fixture',script)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    monkeypatch.setattr(runner,'BUDGETS',(2,4))
    monkeypatch.setattr(runner,'history_at',lambda curriculum,step:32)
    monkeypatch.setattr(runner,'domain',lambda split:'dev029-resume-fixture-'+split)
    monkeypatch.setattr(runner.subprocess,'check_output',lambda command,**kw:'frozen-fixture-source\n' if command[1]=='rev-parse' else '')
    def fixture_validation(model,smoke):
        cases=make_pair(17,32,32,'retention','at',domain='dev029-resume-fixture-validation')
        state=model.encode(*encode([c.observations for c in cases]))
        q=torch.zeros(2,8,3,14)
        return dict(primary_accuracy=.5,cross_entropy=float(model.decode(state,q).square().mean().detach()),
                    rows=[dict(fingerprint=c.fingerprint) for c in cases])
    monkeypatch.setattr(runner,'validate',fixture_validation)
    common=dict(arm='event_table',learning_rate=.002,curriculum='fixed512',seed=311)
    full=tmp_path/'full';split=tmp_path/'split'
    runner.run(SimpleNamespace(**common,until=4,resume=None,output=str(full)))
    runner.run(SimpleNamespace(**common,until=2,resume=None,output=str(split)))
    runner.run(SimpleNamespace(**common,until=4,resume=str(split/'2.pt'),output=str(split)))
    a=torch.load(full/'4.pt',weights_only=True);b=torch.load(split/'4.pt',weights_only=True)
    assert a['training_fingerprints']==b['training_fingerprints']
    assert a['observed_events']==b['observed_events']
    assert all(torch.equal(a['model'][name],b['model'][name]) for name in a['model'])
    assert a['optimizer']['param_groups']==b['optimizer']['param_groups']
    for index,state in a['optimizer']['state'].items():
        assert all(torch.equal(value,b['optimizer']['state'][index][name]) for name,value in state.items())
