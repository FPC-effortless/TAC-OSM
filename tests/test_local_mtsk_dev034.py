import copy
import pytest
import torch
from tac_osm.development.local_mtsk import ARMS,BankedMTSK,LocalMTSKPredictor


def test_augmented_state_budget_and_responsibility_normalization():
    m=BankedMTSK();x=torch.randn(2,12);s=m.initial(x)
    assert s[0].numel()==768
    matrix,responsibility=m.unpack(s)
    assert matrix[0].numel()==576 and responsibility[0].numel()==192
    assert torch.allclose(responsibility.sum(-2),torch.ones(2,8,3))


def test_unrelated_updates_preserve_whole_bank_for_503_events():
    torch.set_num_threads(1);m=BankedMTSK();x=torch.randn(2,12)
    s=m.update(m.initial(x),x,torch.tensor([3,5]));saved=s.clone()
    with torch.no_grad():
        for _ in range(503):s=m.update(s,x,torch.tensor([1,2]))
    assert torch.equal(s[0,3],saved[0,3]) and torch.equal(s[1,5],saved[1,5])


def test_batch_addresses_and_rejected_whole_transition():
    m=BankedMTSK();x=torch.randn(2,12);s=torch.randn(2,8,96)
    _,r=m.unpack(s);r.copy_(r.softmax(-2))
    out=m.update(s,x,torch.tensor([3,5]),torch.tensor([True,False]))
    assert not torch.equal(out[0,3],s[0,3]);assert torch.equal(out[1],s[1])
    assert torch.equal(out[0,[0,1,2,4,5,6,7]],s[0,[0,1,2,4,5,6,7]])


def test_all_rejected_commit_skips_proposal_and_is_identity():
    m=BankedMTSK();x=torch.randn(2,12);s=m.initial(x)
    m.proposal=lambda *args: (_ for _ in ()).throw(AssertionError('rejected proposal executed'))
    assert m.update(s,x,torch.tensor([0,1]),torch.tensor([False,False])) is s


def test_nonfinite_observation_proposal_fails_and_explicit_commit_rejects():
    m=BankedMTSK();x=torch.randn(2,12);s=m.initial(x)
    m.proposal=lambda old,x:torch.full_like(old,float('nan'))
    with pytest.raises(FloatingPointError):m.update(s,x,torch.tensor([0,1]))
    assert torch.equal(m.update(s,x,torch.tensor([0,1]),torch.tensor([True,True])),s)


def test_new_state_derivative_is_confined_to_addressed_bank():
    m=BankedMTSK();x=torch.randn(1,12);s=m.initial(x).requires_grad_()
    result=m.update(s,x,torch.tensor([3]));result[:,3].sum().backward()
    assert s.grad[:,3].abs().sum()>0
    assert s.grad[:,[0,1,2,4,5,6,7]].abs().sum()==0


def test_relabeling_state_address_and_embedding_preserves_kernel_transition():
    m=BankedMTSK();other=copy.deepcopy(m);x=torch.randn(2,12);s=m.initial(x)
    permutation=torch.tensor([3,0,7,1,6,2,5,4]);inverse=permutation.argsort()
    with torch.no_grad():other.bank_embedding.copy_(m.bank_embedding[permutation])
    a=m.update(s,x,torch.tensor([3,5]))
    b=other.update(s[:,permutation],x,inverse[torch.tensor([3,5])])
    assert torch.equal(a[:,permutation],b)


def test_global_control_changes_unaddressed_banks():
    m=BankedMTSK(local=False);x=torch.randn(2,12);s=m.initial(x)
    out=m.update(s,x,torch.tensor([0,1]))
    assert all(not torch.equal(out[:,e],s[:,e]) for e in range(8))


def test_predictor_shapes_read_only_queries_and_shared_initial_heads():
    torch.set_num_threads(1);models=[LocalMTSKPredictor(a,34311) for a in ARMS]
    for model in models[1:]:
        for name in ('event_encoder','query_encoder','head','adapter'):
            assert all(torch.equal(a,b) for a,b in zip(getattr(models[0],name).parameters(),getattr(model,name).parameters()))
    events=torch.randn(2,4,16);entities=torch.tensor([[0,1,2,3],[4,5,6,7]]);values=torch.zeros(2,4,dtype=torch.long)
    state=models[-1].encode(events,entities,values);before=state.clone()
    out=models[-1].decode(state,torch.randn(2,8,3,14))
    assert out.shape==(2,8,3,2) and torch.equal(state,before)
    out.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in models[-1].parameters() if p.grad is not None)


def test_invalid_addresses_and_reset_independence():
    m=BankedMTSK();x=torch.randn(2,12);s=m.initial(x)
    with pytest.raises(ValueError):m.update(s,x,torch.tensor([8,0]))
    with pytest.raises(ValueError):m.update(s,x,torch.tensor([1.,2.]))
    m.update(s,x,torch.tensor([1,2]))
    assert torch.equal(s,m.initial(x))
