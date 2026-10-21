"""Native feature extraction and capacity-controlled cross-fitted residual probes."""
import torch
from torch import nn
from tac_osm.me028_nonlinear_readout import NativeNonlinearObservableReadout


class CrossfitNativeEncoder(NativeNonlinearObservableReadout):
    """Same native parameters; additionally export causal prefix features."""
    @torch.jit.export
    def prefix_features(self, items: torch.Tensor, meta: torch.Tensor,
                        action: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.emb(items), meta), -1)
        state = x.new_zeros((items.size(0), 64))
        out = torch.jit.annotate(list[torch.Tensor], [])
        for t in range(128):
            state = self.advance(x[:, t], state)
            k = t + 1
            if k == 32 or k == 64 or k == 96:
                common = self.observables(items, meta, items[:, k], meta[:, k, 0:1], k)
                prior = (meta[:, k-32:k, 1].sum(-1, keepdim=True)+1.)/34.
                out.append(torch.cat((self.features(state), common, torch.logit(prior)), -1))
        common = self.observables(items, meta, action, duration, 128)
        prior = (meta[:, -32:, 1].sum(-1, keepdim=True)+1.)/34.
        out.append(torch.cat((self.features(state), common, torch.logit(prior)), -1))
        return torch.stack(out, 1)


class ResidualProbe(nn.Module):
    __constants__ = ['with_state']
    def __init__(self, with_state: bool):
        super().__init__()
        self.with_state = with_state
        self.hidden = nn.Linear(90 if with_state else 26, 13 if with_state else 43)
        self.output = nn.Linear(13 if with_state else 43, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        x = features[..., :90] if self.with_state else features[..., 64:90]
        return offset + self.output(torch.tanh(self.hidden(x))).squeeze(-1)


def fold_ids(users: torch.Tensor) -> torch.Tensor:
    return torch.div(users, 16, rounding_mode='floor').remainder(3)


def role_indices(users: torch.Tensor, rotation: int):
    if rotation not in (0, 1, 2):
        raise ValueError('unregistered rotation')
    if users.ndim != 1 or bool((users.remainder(16) != 0).any()):
        raise ValueError('only original user%16==0 cohort')
    fold = fold_ids(users)
    return tuple(torch.nonzero(fold == k).flatten()
                 for k in (rotation, (rotation+1)%3, (rotation+2)%3))
