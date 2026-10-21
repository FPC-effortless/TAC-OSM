"""Mathematical/causal checks only; generated tensors are not capability evidence."""
import pytest
torch=pytest.importorskip('torch')
from tac_osm.me034_stable_inputs import COUNTS,build_model,ProjectedGRU,EWMABank
from tac_osm.me032_conditional_memory import CommonHead


def model(arm):
    torch.manual_seed(7)
    return build_model(arm,CommonHead().state_dict(),torch.randn(16000,8),34001,
        torch.full((16000,),.5),torch.zeros(12),torch.ones(12),0.)


def inputs():
    g=torch.Generator().manual_seed(8)
    return (torch.randint(16000,(2,128),generator=g),torch.rand(2,128,3,generator=g),
        torch.randint(16000,(2,4),generator=g),torch.randn(2,4,28,generator=g))


@pytest.mark.parametrize('arm',COUNTS)
def test_active_parameters_and_script(arm):
    n=model(arm);x=inputs();s=torch.jit.script(n)
    assert sum(p.numel() for p in n.parameters() if p.requires_grad)==COUNTS[arm]
    torch.testing.assert_close(s(*x),n(*x),rtol=0,atol=1e-7)


def test_frozen_native_identical_initialization_to_adapted():
    frozen,adapted=model('native_frozen'),model('native_adapt')
    assert all(torch.equal(v,adapted.state_dict()[k]) for k,v in frozen.state_dict().items())
    assert not frozen.memory.core.emb.weight.requires_grad
    assert adapted.memory.core.emb.weight.requires_grad
    torch.testing.assert_close(frozen(*inputs()),adapted(*inputs()),rtol=0,atol=0)


def test_gru_fixed_coordinates_injective_and_full_gradient_path():
    n=model('gru_frozen');q=n.memory.coordinates
    torch.testing.assert_close(q.T@q,torch.eye(36),rtol=0,atol=5e-7)
    assert not q.requires_grad and q.shape==(64,36)
    # Centered output starts at zero: enable it to test all real body paths.
    with torch.no_grad():n.correction.direct.weight.fill_(.01)
    n(*inputs()).square().mean().backward()
    for name,p in n.memory.named_parameters():
        if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name


def test_gru_does_not_read_unobserved_future_feedback():
    n=model('gru_frozen').memory;x=inputs()
    a=n(*x[:3]);changed=x[1].clone();changed[:,32:]+=3
    b=n(x[0],changed,x[2]);torch.testing.assert_close(a[:,0],b[:,0],rtol=0,atol=0)


def test_current_only_ignores_all_personal_history_channels():
    n=model('current_only')
    with torch.no_grad():n.output.weight.fill_(.1)
    x=inputs();a=n(*x);c=x[3].clone();c[...,9:14]+=100;c[...,15:]+=100
    b=n(x[0].flip(1),x[1]+100,x[2],c)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_ewma_features_equal_original_me021_at_final_prefix():
    import numpy as np
    from scripts.me021_primary_interaction import features
    x=inputs();meta=x[1].clone();meta[:,:,1]=(meta[:,:,1]>.5).float()
    c=x[3].clone();c[:,:,27]=.7
    n=model('ewma_mlp')
    a=n.features(x[0],meta,x[2],c)[:,-1].numpy()
    data={'items':x[0],'metadata':meta,'action':x[2][:,-1],'duration':torch.full((2,1),.7)}
    b=features(data,n.popularity.numpy())
    np.testing.assert_allclose(a,b,rtol=0,atol=2e-7)


def test_dense_arithmetic_and_active_parameter_parity():
    assert abs(COUNTS['gru_frozen']/COUNTS['native_frozen']-1)<.01
    native=128*(11*32+32*32+43*32+2*64*32)
    gru=128*(11*24+3*36*(24+36))+4*36*64
    assert native==876544 and gru==872448
    assert abs(gru/native-1)<.01


def test_registered_protocol_and_final_remainder():
    from scripts.me034_stable_research import PROTOCOL,FINAL_MOD,auxiliary
    assert FINAL_MOD==6 and 'Bonferroni97.5%' in PROTOCOL.read_text()
    x=inputs();d={'metadata':x[1],'duration':torch.tensor([[.2],[.3]])}
    c=auxiliary(d,x[3][...,:27])
    assert c.shape==(2,4,28)
    assert torch.equal(c[:,-1,27],d['duration'].flatten())
