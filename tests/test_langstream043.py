"""Constructed original-graph execution tests, not trained utility evidence."""
from dataclasses import replace
from pathlib import Path
import sys
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'research/mathematical-engineering/langstream043'),str(ROOT/'research/mathematical-engineering/langlayer040'),str(ROOT/'research/mathematical-engineering/langmemory042')]
from incremental import IncrementalTAC043,SHAPES
from factorial_depth import FactorialDepthControl040
from candidate import controlled_position


def model(dtype=torch.float64):
    torch.manual_seed(43043)
    return FactorialDepthControl040(state_width=48,upper_graph=False).to(dtype).eval()


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('mode',['native','cap_all'])
def test_complete_prefix_states_masks_and_bulk_prefill(dtype,mode):
    m=model(dtype);runner=IncrementalTAC043(m,mode);owners=('first','second')
    tol={'atol':3e-5,'rtol':1e-4} if dtype==torch.float32 else {'atol':2e-10,'rtol':2e-8}
    for length in (1,7,17,129,513):
        x=torch.randint(0,256,(2,length))
        for alternating in (False,True):
            mask=torch.ones_like(x,dtype=torch.bool)
            if alternating: mask=(torch.arange(length)[None,:]+torch.arange(2)[:,None])%2==0
            with torch.no_grad(),controlled_position(m,mode): expected,tr=m(x,write_mask=mask,return_trace=True)
            state=runner.start(owners);outputs=[]
            for t in range(length):
                snapshot={n:v.clone() for n,v in state.tensors.items()}
                y,next_state=runner.step(x[:,t],state,owners=owners,write_mask=mask[:,t]);outputs.append(y)
                assert all(torch.equal(v,state.tensors[n]) for n,v in snapshot.items())
                assert torch.allclose(next_state.tensors['native'],tr['native_memory'][:,t],**tol)
                assert next_state.tensor_bytes()==2*976*torch.tensor([],dtype=dtype).element_size()
                state=next_state
            assert torch.allclose(torch.stack(outputs,1),expected,**tol)
            y,bulk=runner.prefill(x,owners=owners,write_mask=mask)
            assert torch.equal(y,expected[:,-1])
            for name in SHAPES: assert torch.allclose(bulk.tensors[name],state.tensors[name],**tol),name
            continuation=torch.tensor([17,23])
            a,st=runner.step(continuation,bulk,owners=owners)
            joined=torch.cat((x,continuation[:,None]),1);joined_mask=torch.cat((mask,torch.ones(2,1,dtype=torch.bool)),1)
            with torch.no_grad(),controlled_position(m,mode): b=m(joined,write_mask=joined_mask)[:,-1]
            assert torch.allclose(a,b,**tol)


def test_owners_reset_chunks_and_exact_snapshot_restart(tmp_path):
    m=model();r=IncrementalTAC043(m);owners=('doc-A','doc-B');x=torch.randint(0,256,(2,17))
    _,state=r.prefill(x[:,:7],owners=owners)
    path=tmp_path/'state.pt';r.save_state(state,path)
    restored=r.load_state(path,owners=owners)
    with pytest.raises(FileExistsError): r.save_state(state,path)
    for t in range(7,17):
        y,state=r.step(x[:,t],state,owners=owners)
        z,restored=r.step(x[:,t],restored,owners=owners)
        assert torch.equal(y,z)
        assert all(torch.equal(state.tensors[k],restored.tensors[k]) for k in SHAPES)
    with pytest.raises(ValueError,match='owner'): r.step(x[:,0],state,owners=('doc-B','doc-A'))
    with pytest.raises(ValueError,match='owner'): r.load_state(path,owners=('other','doc-B'))
    other=model();other.ffn_out.bias.data[0]+=1
    with pytest.raises(ValueError,match='model/mode'): IncrementalTAC043(other).load_state(path,owners=owners)
    with pytest.raises(ValueError,match='model/mode'): IncrementalTAC043(m,'cap_all').load_state(path,owners=owners)
    single=IncrementalTAC043(m);_,a=single.prefill(x[:1],owners=('doc-A',))
    assert torch.allclose(y[:1],a.tensors['last_logits'],atol=1e-10,rtol=1e-8)
    r.save_state(state,tmp_path/'final.pt')
    payload=torch.load(tmp_path/'final.pt',weights_only=True);payload['tensors']['native'][0,0,0]+=1
    torch.save(payload,tmp_path/'tampered.pt')
    with pytest.raises(ValueError,match='integrity'): r.load_state(tmp_path/'tampered.pt',owners=owners)


def test_fail_closed_state_model_and_token_boundaries():
    m=model();r=IncrementalTAC043(m);state=r.start(('one',));x=torch.tensor([3])
    for token in (torch.tensor([-1]),torch.tensor([256]),torch.tensor([2.]),torch.tensor([[2]])):
        with pytest.raises(ValueError): r.step(token,state,owners=('one',))
    for bad in (replace(state,clock=-1),replace(state,clock=2**63-1),replace(state,tensors={}),replace(state,position_mode='other')):
        with pytest.raises(ValueError): r.step(x,bad,owners=('one',))
    with pytest.raises(ValueError): r.start(('same','same'))
    with pytest.raises(ValueError): IncrementalTAC043(m.train())
    m.eval()
    with torch.no_grad(): m.ffn_out.bias.add_(.1)
    with pytest.raises(ValueError,match='weights'): r.step(x,state,owners=('one',))


def test_full_state_probe_is_pure_and_gradient_scope_explicit():
    from stream_probe import adapter
    from tac_osm.research_probe import probe_memory,probe_dependencies
    m=model();x=torch.randint(0,256,(2,7));a=adapter()
    sample={'tokens':x,'owners':('a','b'),'write_mask':torch.ones_like(x,dtype=torch.bool)}
    result=probe_memory(m,sample,a)
    assert result['query_does_not_write_state']
    assert len(result['state_components'])==9
    assert probe_dependencies(m,sample,a)['status']=='INCONCLUSIVE_UNINSTRUMENTED'
