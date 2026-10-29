"""Engineering fixtures; these do not measure trained memory quality."""
from pathlib import Path
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'research/mathematical-engineering/langmemory042'),str(ROOT/'research/mathematical-engineering/context026')]
from candidate import transform,controlled_position
from context_transition import ContextTransition026,phi_position


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_transform_identity_in_training_range_and_does_not_mutate(dtype):
    f=phi_position(torch.arange(513),dtype);original=f.clone()
    for mode in ('native','cap_log','cap_all'):
        y=transform(f,mode)
        assert torch.equal(y[:128],f[:128])
        assert torch.equal(f,original)
    assert torch.equal(transform(f,'cap_log')[128:,1:],f[128:,1:])
    assert torch.equal(transform(f,'cap_all')[128:],f[127].expand(385,7))


def test_model_short_identity_prefix_and_hook_removal_on_failure():
    torch.manual_seed(42042);m=ContextTransition026().eval();x=torch.randint(0,256,(1,129))
    future=x.clone();future[:,100:]^=255
    with torch.no_grad():
        native=m(x)
        for mode in ('cap_log','cap_all'):
            with controlled_position(m,mode):
                a=m(x);b=m(future)
            assert torch.equal(a[:,:128],native[:,:128])
            assert torch.equal(a[:,:100],b[:,:100])
        try:
            with controlled_position(m,'cap_log'): raise RuntimeError('forced fixture')
        except RuntimeError: pass
        assert torch.equal(m(x),native)
    assert not m.pos._forward_pre_hooks
    with pytest.raises(ValueError): transform(torch.zeros(1,7),'other')
