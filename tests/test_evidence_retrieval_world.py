import torch
import pytest
from tac_osm.variants.evidence_retrieval_world import EvidenceRetrievalWorld

torch.set_num_threads(1)

def test_stream_chunks_prefix_and_gradients():
    torch.manual_seed(42)
    m=EvidenceRetrievalWorld().double()
    assert 95000 <= sum(p.numel() for p in m.parameters()) <= 105000
    x=torch.randn(2,13,14,dtype=torch.float64,requires_grad=True)
    u=torch.randn_like(x,requires_grad=True)
    valid=torch.ones(2,13,dtype=torch.bool);valid[0,3:5]=False
    y,s,tr=m(x,u,valid=valid,return_trace=True)
    st=None; ys=[]
    for start,end in [(0,3),(3,8),(8,13)]:
        z,st=m(x[:,start:end],u[:,start:end],valid=valid[:,start:end],state=st,return_state=True)
        ys.append(z)
    torch.testing.assert_close(y,torch.cat(ys,1),rtol=1e-10,atol=1e-11)
    for a,b in zip((s.basis_sum,s.memory),(st.basis_sum,st.memory)):
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)
    g1=torch.autograd.grad(y.square().sum(),tuple(m.parameters())+(x,u),retain_graph=True)
    g2=torch.autograd.grad(torch.cat(ys,1).square().sum(),tuple(m.parameters())+(x,u))
    for a,b in zip(g1,g2):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-10)
    assert all(g.norm()>0 for g in g1[:-2])
    x2=x.detach().clone();x2[:,8:]+=50
    torch.testing.assert_close(y[:,:8],m(x2,u.detach(),valid=valid)[:,:8],rtol=0,atol=1e-12)

def test_padding_identity_and_state_checks():
    m=EvidenceRetrievalWorld();s=m.initial_state(2)
    x=torch.randn(2,7,14);u=torch.randn_like(x)
    y,z=m(x,u,valid=torch.zeros(2,7,dtype=torch.bool),state=s,return_state=True)
    assert torch.equal(y,x) and torch.equal(z.memory,s.memory)
    assert torch.equal(z.count,s.count) and torch.equal(z.basis_sum,s.basis_sum)
    with pytest.raises(ValueError):m(x,u,valid=torch.ones(2,7))


def test_one_candidate_for_all_six_families():
    from tac_osm.variants.parallel_six import instantiate_untrained_core, PARALLEL_FAMILIES, require_qualified_domain_head
    for name in PARALLEL_FAMILIES:
        m=instantiate_untrained_core(name,state_dim=14,action_dim=14,candidate='evidence_retrieval')
        assert type(m) is EvidenceRetrievalWorld
        assert sum(p.numel() for p in m.parameters())==95874
        with pytest.raises(RuntimeError):require_qualified_domain_head(name)
