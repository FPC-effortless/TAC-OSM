"""SENSE-TOK-003: source-only, never calls optimizer.step."""
import random
import torch
import pytest
from tac_osm.causal_byte_probe import predecision_examples, CausalByteGRU
from tac_osm.patch_sparse_byte_probe import PatchSparseByteGRU, sparse_parameter_receipt
from tac_osm.causal_patch_stream import STREAM_VARIANTS


def test_01_exact_size_matching_from_live_model():
    m=PatchSparseByteGRU()
    p=sparse_parameter_receipt(m)
    assert p['trainable_parameters']==p['deployed_learned_parameters']==100544
    assert p['100k_absolute_band']
    assert p['baseline_100400_relative_gap'] < .01
    assert sum(x.numel() for x in CausalByteGRU().parameters())==100400
    assert p['training_authorized'] is False


@pytest.mark.parametrize('variant', STREAM_VARIANTS)
def test_02_forward_shape_recurrent_updates_and_no_eos(variant):
    model=PatchSparseByteGRU().eval()
    raw=(b'abc123; def456\n' * 5)
    ids,_,targets=predecision_examples(raw,variant)
    with torch.no_grad():
        outputs, (r,)=model(ids[None],variant=variant,audit=True)
    assert outputs.shape==(1,len(raw),256)
    assert torch.isfinite(outputs).all()
    assert r.byte_readouts==len(raw)
    assert r.observed_bytes==len(raw)-1
    assert r.recurrent_updates<len(raw)-1
    assert r.recurrent_updates==len(r.committed_patch_ends)
    assert r.recurrent_updates==len(r.recurrent_update_times)
    assert all(0<=end<=time<=len(raw)-1 for end,time in zip(r.committed_patch_ends,r.recurrent_update_times))
    assert not r.eos_used


@pytest.mark.parametrize('variant', STREAM_VARIANTS)
def test_03_actual_output_future_poison_invariance(variant):
    torch.manual_seed(55)
    model=PatchSparseByteGRU().eval()
    observed=b'abc123\n' * 2
    xa=observed+b'!!! future only\n'
    xb=observed+b'999 future only\n'
    a,_,_=predecision_examples(xa,variant)
    b,_,_=predecision_examples(xb,variant)
    with torch.no_grad():
        pa=model(a[None],variant=variant)
        pb=model(b[None],variant=variant)
    assert torch.equal(pa[:,:len(observed)+1],pb[:,:len(observed)+1])


@pytest.mark.parametrize('variant',STREAM_VARIANTS)
def test_04_all_parameters_have_finite_gradient_no_fit(variant):
    torch.manual_seed(771)
    model=PatchSparseByteGRU().train()
    raw=b'abc123 !!! def456 789\n' * 3
    x,_,y=predecision_examples(raw,variant)
    logits=model(x[None],variant=variant)
    loss=torch.nn.functional.cross_entropy(logits[0],y)
    assert torch.isfinite(loss)
    loss.backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        assert p.grad.abs().sum().item()>0,name


def test_05_segmentation_is_a_causal_component_of_output():
    torch.manual_seed(448)
    model=PatchSparseByteGRU().eval()
    raw=b'abc123:def456; xyz789!=\n' * 3
    ids,_,_=predecision_examples(raw,'fixed')
    with torch.no_grad():
        t2,a=model(ids[None],variant='sensory',audit=True)
        t2b,b=model(ids[None],variant='sensory_v2',audit=True)
        t0,c=model(ids[None],variant='fixed',audit=True)
    assert a[0].committed_patch_ends!=b[0].committed_patch_ends
    assert b[0].recurrent_updates!=c[0].recurrent_updates
    assert not torch.equal(t2,t2b)
    assert not torch.equal(t0,t2)


def test_06_empty_zero_batch_and_invalid_data():
    m=PatchSparseByteGRU()
    empty=torch.empty((1,0),dtype=torch.long)
    o=m(empty,variant='fixed')
    assert o.shape==(1,0,256)
    z=m(torch.empty((0,0),dtype=torch.long),variant='fixed')
    assert z.shape==(0,0,256)
    with pytest.raises(ValueError):
        m(torch.tensor([[65,66]]),variant='fixed')
    with pytest.raises(ValueError):
        m(torch.tensor([[256,256]]),variant='fixed')
    with pytest.raises(ValueError):
        m(torch.tensor([[256,66]]),variant='invalid')


def test_07_batches_reset_state_and_replay_is_deterministic():
    model=PatchSparseByteGRU().eval()
    x,_,_=predecision_examples(b'abc123-\n'*3,'sensory_v2')
    y,_,_=predecision_examples(b'xyz999+\n'*3,'sensory_v2')
    with torch.no_grad():
        batched,r=model(torch.stack([x,y]),variant='sensory_v2',audit=True)
        single=model(x[None],variant='sensory_v2')
    assert torch.equal(batched[0], single[0])
    assert r[0].observed_bytes==len(x)-1
    assert r[1].observed_bytes==len(y)-1


def test_08_self_predicting_byte_not_visible_yet():
    torch.manual_seed(700)
    model=PatchSparseByteGRU().eval()
    with torch.no_grad():
        # Both models condition on exactly the same *empty* prefix at index 0.
        a,_,_=predecision_examples(b'abc1','sensory_v2')
        b,_,_=predecision_examples(b'abcA','sensory_v2')
        u=model(a[None],variant='sensory_v2')
        v=model(b[None],variant='sensory_v2')
        assert torch.equal(u[:,:4],v[:,:4])


@pytest.mark.parametrize('variant', STREAM_VARIANTS)
def test_09_prefix_only_evaluation_exact_replay(variant):
    torch.manual_seed(91)
    model=PatchSparseByteGRU().eval()
    raw=b'abc123 <> def456\n' * 2
    x,_,_=predecision_examples(raw,variant)
    with torch.no_grad():
        whole=model(x[None],variant=variant)
        for t in (1,2,3,5,7,12,20,len(x)):
            sliced=model(x[None,:t],variant=variant)
            assert torch.equal(sliced,whole[:,:t]),(variant,t)
