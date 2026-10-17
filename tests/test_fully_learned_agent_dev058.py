import ast
from pathlib import Path
import sys
import numpy as np
import torch
from tac_osm.development.learned_agent import LearnedAgent
from tac_osm.development.learned_agent_world import World,snapshots,outcomes


def public_case(seed=8581001):
    w=World(seed,0,4,32)
    descriptors,valid,queries,qvalid,_=w.candidates()
    return w,[torch.from_numpy(x)[None] for x in (descriptors,valid,queries,qvalid)]


def test_runtime_module_has_only_torch_and_standard_library_imports():
    path=Path(__file__).resolve().parents[1]/'src/tac_osm/development/learned_agent.py'
    tree=ast.parse(path.read_text())
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.extend(x.name.split('.')[0] for x in node.names)
        elif isinstance(node,ast.ImportFrom):imports.append(node.module.split('.')[0])
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):assert node.func.attr!='argmin'
    assert set(imports)<={'dataclasses','math','torch'}
    assert 'teacher' not in LearnedAgent.forward.__code__.co_names
    assert 'outcomes' not in LearnedAgent.forward.__code__.co_names


def test_closed_neural_admission_is_identity_and_open_routes_by_neural_address():
    torch.set_num_threads(1);agent=LearnedAgent(seed=8581002).eval()
    state=agent.initial(2);event=torch.randn(2,22)
    with torch.no_grad():
        for p in agent.admission.parameters():p.zero_()
        agent.admission[-1].bias.fill_(-30)
        after,trace=agent.observe(state,event)
        assert trace['writes']==0
        for a,b in zip(state,after):torch.testing.assert_close(a,b,rtol=0,atol=0)
        agent.admission[-1].bias.fill_(30)
        for p in agent.address.parameters():p.zero_()
        agent.address[-1].bias[7]=30
        after,trace=agent.observe(state,event)
        assert trace['address'].tolist()==[7,7]
        torch.testing.assert_close(after[0][:,7],agent.encoder(event))
        assert after[1][:,7].tolist()==[1,1]


def test_candidate_order_is_equivariant_and_state_is_only_opaque_tensors():
    agent=LearnedAgent(seed=8581003).eval();w,public=public_case()
    state=agent.initial(1)
    with torch.no_grad():
        for e in w.events:state,_=agent.observe(state,torch.tensor(e)[None])
        original=agent(state,*public)
        permutation=torch.randperm(13)
        changed=agent(state,public[0][:,permutation],public[1][:,permutation],*public[2:])
        torch.testing.assert_close(original['logits'][:,permutation],changed['logits'],rtol=1e-4,atol=1e-5)
        assert all(isinstance(x,torch.Tensor) for x in state)
        assert sum(x.numel()*x.element_size() for x in state)==32*49*4


def test_learning_losses_reach_every_policy_and_world_component():
    scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
    try:
        from run_fully_learned_dev058 import cache,loss_for,writer_fit
        agent=LearnedAgent(seed=8581004)
        writer=writer_fit(agent,8581004,20)
        assert writer['parameter_change_l1']>0
        data,_=cache(agent,snapshots(8581005,8))
        loss,_=loss_for(agent,data,torch.arange(8));loss.backward()
        for group in ('encoder','action_encoder','read_query','read_key','node','graph_q','operator','operator_selector','transition','verifier','future_policy','policy','outcome'):
            assert sum(float(p.grad.abs().sum()) for p in getattr(agent,group).parameters() if p.grad is not None)>0,group
    finally:sys.path.remove(str(scripts))


def test_neural_writer_cache_is_identical_to_streamed_latent_memory():
    scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
    try:
        from run_fully_learned_dev058 import cache
        agent=LearnedAgent(seed=8581006).eval();rows=snapshots(8581007,3)
        data,_=cache(agent,rows)
        with torch.no_grad():
            expected=agent.encode_bank(data['bank'],data['occupied'])
            for i,row in enumerate(rows):
                state=agent.initial(1)
                for e in row['events']:state,_=agent.observe(state,torch.tensor(e)[None])
                for actual,reference in zip(state,expected):torch.testing.assert_close(actual,reference[i:i+1],rtol=1e-5,atol=1e-6)
    finally:sys.path.remove(str(scripts))


def test_public_receipts_cover_all_changed_entities_and_private_targets_stay_outside_model():
    w=World(8581008,0,4,32)
    w.parents=np.array([-1,0,-1,2]);w.bits[:]=0
    receipt=w.act(('repair',0))
    changed=[i for i,(a,b) in enumerate(zip(w.actions[-1]['before'],w.actions[-1]['after'])) if a!=b]
    keys=[int(e[4:12].argmax()) for e in receipt]
    assert all(int(w.keys[e]) in keys for e in changed)
    assert len(outcomes(np.array([1,1]),np.array([-1,0]),('repair',1)))==1


def test_restored_checkpoint_matches_predictions_without_teacher(monkeypatch):
    import tac_osm.development.learned_agent_world as environment
    w,public=public_case(8581009);agent=LearnedAgent(seed=8581010).eval()
    monkeypatch.setattr(environment,'teacher',lambda *a:(_ for _ in ()).throw(AssertionError('teacher called')))
    monkeypatch.setattr(environment,'outcomes',lambda *a:(_ for _ in ()).throw(AssertionError('simulator called')))
    with torch.no_grad():
        state=agent.initial(1)
        for e in w.events:state,_=agent.observe(state,torch.tensor(e)[None])
        expected=agent(state,*public)['logits']
        restored=LearnedAgent(seed=3).eval();restored.load_state_dict(agent.state_dict())
        torch.testing.assert_close(expected,restored(state,*public)['logits'],rtol=0,atol=0)
