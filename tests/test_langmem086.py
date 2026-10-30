"""Constructed boundary/AD engineering checks; no capability evidence or fit."""
from pathlib import Path
import sys
import torch
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/mathematical-engineering/langmem086'))
from compact_tac import CompactDualClippedTAC086

def test_actual100k_all_active_parameters_and_gradients():
    torch.manual_seed(86);m=CompactDualClippedTAC086();torch.set_num_threads(2)
    assert m.counts()=={'trainable':100521,'deployed':100521,'fixed_buffer_bytes':8192}
    x=torch.arange(12).view(1,-1);y=(x+1)%256
    logits,state=m(x,owners=('fixture',))
    torch.nn.functional.cross_entropy(logits.reshape(-1,256),y.flatten()).backward()
    assert state.tensor_bytes()==1920 and state.clock==12
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

def test_streaming_chunk_and_one_byte_prefix_parity_no_inplace_state_write():
    torch.manual_seed(86);m=CompactDualClippedTAC086().eval();tokens=torch.arange(16).view(1,-1)
    with torch.no_grad():
        bulk,bulk_state=m(tokens,owners=('owner',))
        state=m.start(('owner',));before={k:v.clone() for k,v in state.tensors.items()};parts=[]
        for i in range(16):
            out,state=m(tokens[:,i:i+1],state=state,owners=('owner',));parts.append(out)
        assert torch.allclose(bulk,torch.cat(parts,1),atol=3e-5,rtol=3e-5)
        assert all(torch.allclose(bulk_state.tensors[k],state.tensors[k],atol=3e-5,rtol=3e-5) for k in state.tensors)
        initial=m.start(('owner',));m(tokens[:,:1],state=initial,owners=('owner',))
        assert all(torch.equal(initial.tensors[k],before[k]) for k in before)

def test_actual_whole_forward_future_poison_and_entity_isolation():
    torch.manual_seed(86);m=CompactDualClippedTAC086().eval();x=torch.arange(16).view(1,-1)
    poison=x.clone();poison[:,9:]=(poison[:,9:]+93)%256
    with torch.no_grad():
        a,_=m(x,owners=('owner',));b,_=m(poison,owners=('owner',))
        assert torch.equal(a[:,:9],b[:,:9])
        together,_=m(torch.cat((x,poison)),owners=('one','two'))
        assert torch.allclose(together[0],a[0],atol=3e-5,rtol=3e-5)
    with pytest.raises(ValueError):m(x[:,:1],state=m.start(('owner',)),owners=('other',))

def test_no_external_key_or_target_in_inference_interface():
    m=CompactDualClippedTAC086();x=torch.arange(8).view(1,-1)
    with pytest.raises(TypeError):m(x,historical_key=torch.ones(8))
    with pytest.raises(TypeError):m(x,target_byte=42)
    with pytest.raises(ValueError):m(x,lesion='unregistered')
