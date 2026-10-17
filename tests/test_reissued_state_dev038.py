"""DEV-038 probe training contract on a separable held-out toy set."""
import torch
from scripts.probe_reissued_state_dev038 import probe_fit


def test_probe_fit_is_reproducible_and_uses_training_statistics():
    x=torch.tensor([[-2.],[-1.],[1.],[2.]],dtype=torch.float32)
    y=torch.tensor([0.,0.,1.,1.])
    first,mean,scale=probe_fit(x,y)
    second,mean2,scale2=probe_fit(x,y)
    assert torch.equal(first.weight,second.weight) and torch.equal(first.bias,second.bias)
    assert torch.equal(mean,mean2) and torch.equal(scale,scale2)
    with torch.no_grad():pred=(first((x-mean)/scale).flatten()>=0).float()
    assert torch.equal(pred,y)
