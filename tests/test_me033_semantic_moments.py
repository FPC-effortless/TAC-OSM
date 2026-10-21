"""Causal information paths and checkpoint-fixed streaming identities only."""
import pytest
import torch
from tac_osm.me033_semantic_moments import SemanticMoments,COUNTS,build_model
from tac_osm.me032_conditional_memory import CommonHead


def data():
    torch.set_num_threads(1);torch.manual_seed(33033)
    items=torch.randint(16000,(2,128));meta=torch.rand(2,128,3)
    offered=torch.cat((items[:,[32,64,96]],torch.randint(16000,(2,1))),-1)
    return items,meta,offered


@pytest.mark.parametrize('query',[False,True])
def test_stream_merge_prefixes_match_bulk_and_gradients(query):
    items,meta,offered=data();net=SemanticMoments(query).double();meta=meta.double()
    script=torch.jit.script(net);state=net.empty(meta);left=net.empty(meta);right=net.empty(meta)
    assert state.shape==(2,4 if query else 12)
    checkpoints=[]
    for t in range(128):
        state=net.observe(state,items[:,t],meta[:,t])
        if t<63:left=net.observe(left,items[:,t],meta[:,t])
        else:right=net.observe(right,items[:,t],meta[:,t])
        if (t+1)%32==0:checkpoints.append(net.project(state,offered[:,(t+1)//32-1]))
    bulk=net(items,meta,offered);stream=torch.stack(checkpoints,1)
    torch.testing.assert_close(bulk,stream,rtol=0,atol=2e-12)
    torch.testing.assert_close(net.merge(left,right),state,rtol=0,atol=2e-12)
    torch.testing.assert_close(script(items,meta,offered),bulk)
    g1=torch.autograd.grad(bulk.square().sum(),net.emb.weight,retain_graph=True)[0]
    g2=torch.autograd.grad(stream.square().sum(),net.emb.weight)[0]
    torch.testing.assert_close(g1,g2,rtol=0,atol=1e-10)


def test_information_treatments_are_real_not_zero_head_vacuous():
    items,meta,offered=data();h=SemanticMoments(False);q=SemanticMoments(True)
    changed=items.clone();changed[:,:]=torch.randint(16000,items.shape)
    assert not torch.allclose(h(items,meta,offered),h(changed,meta,offered))
    assert torch.equal(q(items,meta,offered),q(changed,meta,offered))
    changed_o=torch.randint(16000,offered.shape)
    assert torch.equal(h(items,meta,offered),h(items,meta,changed_o))
    assert not torch.allclose(q(items,meta,offered),q(items,meta,changed_o))
    perm=torch.randperm(128)
    torch.testing.assert_close(h(items,meta,offered)[:,-1],h(items[:,perm],meta[:,perm],offered)[:,-1],atol=2e-6,rtol=0)


@pytest.mark.parametrize('arm',list(COUNTS))
def test_active_counts_teacher_initialization_and_updates(arm):
    items,meta,offered=data();ctx=torch.randn(2,4,27)
    base=CommonHead();emb=torch.randn(16000,8)
    net=build_model(arm,base.state_dict(),emb,33)
    assert sum(p.numel() for p in net.parameters() if p.requires_grad)==COUNTS[arm]
    expected=ctx[...,26]+base(ctx[...,:26])
    torch.testing.assert_close(net(items,meta,offered,ctx),expected)
    script=torch.jit.script(net)
    torch.testing.assert_close(script(items,meta,offered,ctx),expected)
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.003)
    before={k:v.clone() for k,v in net.state_dict().items()}
    for _ in range(3):
        opt.zero_grad(set_to_none=True);net(items,meta,offered,ctx).square().sum().backward();opt.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in net.parameters() if p.requires_grad)
    if arm!='common_adapt':
        assert all(torch.equal(v,net.common.state_dict()[k]) for k,v in base.state_dict().items())
        name='memory.core.emb.weight' if arm=='native_adapt' else 'memory.emb.weight'
        assert torch.equal(before[name],net.state_dict()[name])==('frozen' in arm)


@pytest.mark.parametrize('arm',list(COUNTS))
def test_prefix_noninterference_after_training_path_activated(arm):
    items,meta,offered=data();ctx=torch.randn(2,4,27)
    model=build_model(arm,CommonHead().state_dict(),torch.randn(16000,8),33)
    if arm!='common_adapt':
        with torch.no_grad():model.correction.output.weight.normal_();model.correction.direct.weight.normal_()
    original=model(items,meta,offered,ctx)
    for k,j in ((32,0),(64,1),(96,2)):
        i=items.clone();m=meta.clone();i[:,k:]=torch.randint(16000,i[:,k:].shape)
        m[:,k:]=torch.rand_like(m[:,k:])
        torch.testing.assert_close(model(i,m,offered,ctx)[:,j],original[:,j],rtol=0,atol=1e-6)


@pytest.mark.parametrize('arm',['history_frozen','query_frozen'])
def test_shared_frozen_runtime_preserves_forecast_and_removes_duplicate_table(arm):
    from tac_osm.me032_conditional_memory import FrozenContext
    from tac_osm.me033_shared_runtime import SharedFrozenForecaster
    items,meta,offered=data();emb=torch.randn(16000,8)
    context=FrozenContext(emb,.5,.2,torch.full((16000,),.4),torch.zeros(26),torch.ones(26))
    student=build_model(arm,CommonHead().state_dict(),emb,33)
    with torch.no_grad():student.correction.output.weight.normal_();student.correction.direct.weight.normal_()
    action=offered[:,-1];duration=torch.rand(2,1)
    shared=SharedFrozenForecaster(context,student)
    # The constructor leaves originals distinct and preserves the caller's weights.
    assert context.emb.weight.data_ptr()!=student.memory.emb.weight.data_ptr()
    assert shared.context.emb.weight.data_ptr()==shared.student.memory.emb.weight.data_ptr()
    assert sum(p.numel() for p in shared.parameters())==137977
    original_count=sum(p.numel() for p in context.parameters())+sum(p.numel() for p in student.parameters())
    assert original_count==265977 and (original_count-137977)*4==512000
    expected=student(items,meta,offered,context(items,meta,action,duration))
    assert torch.equal(shared(items,meta,action,duration),expected)
    assert torch.equal(torch.jit.script(shared)(items,meta,action,duration),expected)
    with torch.no_grad():student.memory.emb.weight[0,0]+=1
    with pytest.raises(ValueError):SharedFrozenForecaster(context,student)
