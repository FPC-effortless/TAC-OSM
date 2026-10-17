import pytest
torch = pytest.importorskip("torch")
from tac_osm.development.frozen_action_heads import action_head_loss, frozen_features, predictive_metrics
from tac_osm.development.memory_baselines import MemoryBaseline


@pytest.mark.parametrize("objective", ["forecast", "decision", "contrastive", "joint"])
def test_action_objectives_push_toward_lower_simulator_cost(objective):
    logits = torch.tensor([[0.,0.]],requires_grad=True)
    loss = action_head_loss(logits,torch.tensor([[1.,0.]]),objective)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad[0,0]<0 and logits.grad[0,1]>0


def test_frozen_public_readout_features_have_no_autograd_or_target_input():
    m = MemoryBaseline("schema41",45).eval().requires_grad_(False)
    f,p = frozen_features(m,(torch.zeros(2,3,16),torch.zeros(2,3,dtype=torch.long),torch.zeros(2,3,dtype=torch.long)),torch.zeros(2,8,3,14))
    assert f.shape==(2,8,88) and p.shape==(2,8,2)
    assert not f.requires_grad and not p.requires_grad
    assert not m.head._forward_pre_hooks


def test_calibration_includes_probability_one_and_zero():
    p = torch.tensor([[[0.,1.],[1.,0.]]])
    assert predictive_metrics(p,p)=={"brier":0.,"ece_10":0.}
    assert predictive_metrics(1-p,p)=={"brier":1.,"ece_10":1.}
