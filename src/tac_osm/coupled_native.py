"""Optional Torch native TAC: tied slow feedback and split RMS readout."""
from __future__ import annotations
import math
import torch
from torch import nn


class CoupledNative(nn.Module):
    __constants__ = ["history", "coupling", "rms_readout", "epsilon"]

    def __init__(self, coupling: float, rms_readout: bool, duration_mean: float,
                 duration_scale: float):
        super().__init__()
        if coupling not in (0., .25) or duration_scale <= 0:
            raise ValueError("unregistered coupling/normalization")
        self.history, self.coupling = 128, coupling
        self.rms_readout, self.epsilon = rms_readout, .01
        self.emb = nn.Embedding(16000, 8)
        self.enc = nn.Linear(11, 32)
        self.transition = nn.Linear(32, 32, bias=False)
        self.gate = nn.Linear(43, 32)
        self.slow = nn.Linear(64, 32)
        self.slow_gate = nn.Linear(64, 32)
        self.head = nn.Linear(73, 1)
        nn.init.constant_(self.gate.bias, math.log(1/7))
        nn.init.constant_(self.slow_gate.bias, math.log(1/23))
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.register_buffer("duration_mean", torch.tensor(duration_mean, dtype=torch.float32))
        self.register_buffer("duration_scale", torch.tensor(duration_scale, dtype=torch.float32))

    @torch.jit.export
    def advance_features(self, u: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        z, s = state[:, :32], state[:, 32:]
        recurrent = z+self.coupling*s if self.coupling != 0. else z
        q = torch.tanh(self.enc(u)+self.transition(recurrent))
        g = torch.sigmoid(self.gate(torch.cat((u, z), -1)))
        z = (1.-g)*z+g*q
        context = torch.cat((z, s), -1)
        gs = torch.sigmoid(self.slow_gate(context))
        s = (1.-gs)*s+gs*torch.tanh(self.slow(context))
        return torch.cat((z, s), -1)

    @torch.jit.export
    def stream_step(self, item: torch.Tensor, metadata: torch.Tensor,
                    state: torch.Tensor) -> torch.Tensor:
        return self.advance_features(torch.cat((self.emb(item), metadata), -1), state)

    @torch.jit.export
    def state_features(self, state: torch.Tensor) -> torch.Tensor:
        if not self.rms_readout:
            return state
        z, s = state[:, :32], state[:, 32:]
        z = z/torch.sqrt(z.square().mean(-1, keepdim=True)+self.epsilon)
        s = s/torch.sqrt(s.square().mean(-1, keepdim=True)+self.epsilon)
        return torch.cat((z, s), -1)

    @torch.jit.export
    def readout_features(self, features: torch.Tensor, action: torch.Tensor,
                         duration: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        d = (duration-self.duration_mean)/self.duration_scale
        return self.head(torch.cat((features, self.emb(action), d), -1))+torch.logit(prior)

    @torch.jit.export
    def readout(self, state: torch.Tensor, action: torch.Tensor,
                duration: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        return self.readout_features(self.state_features(state), action, duration, prior)

    @torch.jit.export
    def encode(self, items: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.emb(items), metadata), -1)
        state = x.new_zeros((items.size(0), 64))
        for t in range(self.history):
            state = self.advance_features(x[:, t], state)
        return state

    def forward(self, items: torch.Tensor, metadata: torch.Tensor,
                action: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        prior = (metadata[:, -32:, 1].sum(-1, keepdim=True)+1)/34
        return self.readout(self.encode(items, metadata), action, duration, prior)

    @torch.jit.export
    def prefix_logits(self, items: torch.Tensor, metadata: torch.Tensor,
                      action: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.emb(items), metadata), -1)
        state = x.new_zeros((items.size(0), 64))
        out = torch.jit.annotate(list[torch.Tensor], [])
        for t in range(self.history):
            state = self.advance_features(x[:, t], state)
            depth = t+1
            if depth == 32 or depth == 64 or depth == 96:
                prior = (metadata[:, depth-32:depth, 1].sum(-1, keepdim=True)+1)/34
                out.append(self.readout(state, items[:, depth], metadata[:, depth, 0:1], prior))
        prior = (metadata[:, -32:, 1].sum(-1, keepdim=True)+1)/34
        out.append(self.readout(state, action, duration, prior))
        return torch.cat(out, -1)


@torch.no_grad()
def joint_jacobian(model, u, state):
    """Analytical one-example64x64Jacobian, with local independent partials."""
    z, s = state[:32], state[32:]
    w = model.transition.weight
    q = torch.tanh(model.enc(u)+model.transition(z+model.coupling*s))
    g = torch.sigmoid(model.gate(torch.cat((u, z))))
    a = (torch.diag(1-g)+(g*(1-q.square()))[:, None]*w
         +((q-z)*g*(1-g))[:, None]*model.gate.weight[:, 11:])
    d = (g*(1-q.square()))[:, None]*(model.coupling*w)
    zp = (1-g)*z+g*q
    ctx = torch.cat((zp, s))
    p, gs = torch.tanh(model.slow(ctx)), torch.sigmoid(model.slow_gate(ctx))
    b = ((gs*(1-p.square()))[:, None]*model.slow.weight[:, :32]
         +((p-s)*gs*(1-gs))[:, None]*model.slow_gate.weight[:, :32])
    c = (torch.diag(1-gs)+(gs*(1-p.square()))[:, None]*model.slow.weight[:, 32:]
         +((p-s)*gs*(1-gs))[:, None]*model.slow_gate.weight[:, 32:])
    j = torch.cat((torch.cat((a, d), 1), torch.cat((b@a, c+b@d), 1)), 0)
    return j, a, d, b, c


def rms_jacobian(x, epsilon=.01):
    r = torch.sqrt(x.square().mean()+epsilon)
    return torch.eye(x.numel(), dtype=x.dtype, device=x.device)/r-torch.outer(x, x)/(x.numel()*r**3)
