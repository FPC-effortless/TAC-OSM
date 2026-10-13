"""Mechanism tests for a CED query reader over TAC-OSM's *learned* fixed PLM."""
from __future__ import annotations

import torch
import pytest

from tac_osm.development.finite_ced_memory import (
    FiniteCEDRead, FiniteMemoryCEDModel, copy_common_initialization,
)
from tac_osm.development.protected_memory import ProtectionModel
from tac_osm.development.protection_benchmark import make_pair, tensors


def test_memory_capacity_and_native_shared_initial_weights():
    torch.manual_seed(1)
    native=ProtectionModel(mode="always_update")
    ced=FiniteMemoryCEDModel()
    copy_common_initialization(native,ced)
    for name,v in native.state_dict().items():
        if name in ced.state_dict():
            assert torch.equal(v,ced.state_dict()[name]),name
    assert native.memory.initial(2,torch.zeros(2,12)).shape==(2,4,8,24)
    assert ced.memory.initial(2,torch.zeros(2,12)).numel()==2*768


def test_reader_set_permutation_equivariance_and_dense_all_selected_equivalence():
    torch.manual_seed(5)
    reader=FiniteCEDRead()
    reader.eval()
    views=torch.randn(2,4,32)
    query=torch.randn(2,32)
    perm=torch.tensor([2,0,3,1])
    with torch.no_grad():
        a=reader(views,query)
        b=reader(views[:,perm],query)
        all_k=reader(views,query,second_top_k=4)
    torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(a,all_k,atol=2e-5,rtol=2e-5)


def test_sparse_reuses_same_kv_and_charges_initial_full_scan():
    torch.manual_seed(7)
    reader=FiniteCEDRead()
    views=torch.randn(2,4,32)
    q=torch.randn(2,32)
    reader(views,q)
    dense=reader.last_work
    reader(views,q,second_top_k=2)
    sparse=reader.last_work
    assert dense is not None and sparse is not None
    assert dense.global_kv_scalars==sparse.global_kv_scalars==512
    assert dense.encoder_score_pairs==sparse.encoder_score_pairs==64
    assert dense.first_decoder_score_pairs==sparse.first_decoder_score_pairs==16
    assert dense.second_decoder_score_pairs==16
    assert sparse.second_decoder_score_pairs==8
    assert sparse.memory_state_scalars_per_episode==768


def test_gradient_from_task_decision_to_actual_finite_writer_and_reader():
    torch.manual_seed(11)
    model=FiniteMemoryCEDModel()
    p=make_pair(42,32,4,1.0,.25,"retention")
    a,v,q,c=tensors([p.left,p.right])
    state=model.encode(a,v,trace=False)
    output=model.predict(state,q,c)
    target=torch.tensor([p.left.target,p.right.target])
    loss=torch.nn.functional.cross_entropy(output,target)
    loss.backward()
    assert model.memory.write.weight.grad is not None
    assert model.memory.write.weight.grad.abs().sum()>0
    assert model.ced_read.kv_projection.weight.grad is not None
    assert model.ced_read.kv_projection.weight.grad.abs().sum()>0


def test_opposite_past_only_changes_one_historical_payload():
    pair=make_pair(71,32,4,1.0,.08,"acquisition")
    a,v,q,c=tensors([pair.left,pair.right])
    assert torch.equal(a[0],a[1])
    assert torch.equal(q[0],q[1])
    assert torch.equal(c[0],c[1])
    assert int((v[0]!=v[1]).sum())==1
    assert pair.left.target!=pair.right.target


def test_only_historical_prefix_changes_state_not_future_observations():
    torch.manual_seed(19)
    model=FiniteMemoryCEDModel()
    model.eval()
    pair=make_pair(43,32,4,0.0,.08,"retention")
    a,v,q,c=tensors([pair.left,pair.right])
    with torch.no_grad():
        state=model.encode(a,v,trace=False)
        logits=model.predict(state,q,c)
        # Append hypothetical future events: the *earlier* state and its
        # decision must remain unchanged when computed at the old boundary.
        a_new=torch.cat([a,torch.randn(2,3,12)],dim=1)
        v_new=torch.cat([v,torch.randint(0,2,(2,3))],dim=1)
        augmented=model.encode(a_new,v_new,trace=False)
        assert augmented.shape==state.shape
        old_recomputed=model.encode(a_new[:,:a.shape[1]],v_new[:,:v.shape[1]],trace=False)
        torch.testing.assert_close(state,old_recomputed)
        torch.testing.assert_close(logits,model.predict(old_recomputed,q,c))


def test_bad_shapes_invalid_k_and_nonfinite_fail_closed():
    reader=FiniteCEDRead()
    x=torch.randn(2,4,32)
    q=torch.randn(2,32)
    with pytest.raises(ValueError):
        reader(x,q,second_top_k=5)
    with pytest.raises(ValueError):
        reader(x,torch.randn(2,31))
    with pytest.raises(ValueError):
        reader(x.float().fill_(float("nan")),q)
