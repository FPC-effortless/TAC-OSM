"""Generic probability-linear learned utility, not a physical/action planner.

Utilities and positive cost sensitivity are learned from public experience.
No class has a hard-coded reward, action rule, repair law or stop threshold.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .public_outcome import PublicOutcomeAgent


class ConditionalExpectedValue(nn.Module):
    def __init__(self, outcomes=21, public_features=8, width=64):
        super().__init__(); self.outcomes = outcomes; self.public_features = public_features
        self.utility = nn.Sequential(nn.Linear(public_features, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, outcomes+1))

    def forward(self, features):
        if features.shape[-1] != self.outcomes+1+self.public_features:
            raise ValueError('Declared probabilities, cost and public context required')
        probabilities = features[..., :self.outcomes]
        cost = features[..., self.outcomes:self.outcomes+1]
        context = features[..., self.outcomes+1:]
        values = self.utility(context)
        expected = (probabilities*values[..., :self.outcomes]).sum(-1, keepdim=True)
        return expected+F.softplus(values[..., self.outcomes:])*cost


class SemanticValueAgent(PublicOutcomeAgent):
    def __init__(self, *args, readout='mlp', policy_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        if readout not in ('mlp', 'expected'): raise ValueError(readout)
        self.readout = readout
        torch.manual_seed(policy_seed)
        self.policy = (ConditionalExpectedValue() if readout == 'expected' else
            nn.Sequential(nn.Linear(30, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1)))


def from_parent(weights, spec, parent_seed, policy_seed, readout):
    model = SemanticValueAgent(spec['backend'], spec['width'], spec['aux'], parent_seed,
                               readout=readout, policy_seed=policy_seed).double()
    body = {name: tensor for name, tensor in weights.items() if not name.startswith('policy.')}
    missing, extra = model.load_state_dict(body, strict=False)
    assert not extra and missing and all(name.startswith('policy.') for name in missing)
    for name, parameter in model.named_parameters(): parameter.requires_grad_(name.startswith('policy.'))
    for name, tensor in body.items(): assert torch.equal(model.state_dict()[name], tensor)
    return model
