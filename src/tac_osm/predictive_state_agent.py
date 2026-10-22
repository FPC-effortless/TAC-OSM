"""Untrained integrated TAC candidate; numerical implementation, not capability evidence.

Native fast32/slow32 and 8x8 innovation dynamics follow InnovationMemory at
65a2d549:src/tac_osm/me038_innovation_forecast.py. This standalone module avoids
importing its forecasting-study dependencies. Action readout and masked outcome
correction are prospective extensions. No SSM/attention substitution is used.
The caller owns entity routing, chronology, normalization and outcome availability.
"""
from typing import NamedTuple
import math
import torch
from torch import nn


class Belief(NamedTuple):
    fast: torch.Tensor
    slow: torch.Tensor
    memory: torch.Tensor


class PredictiveStateAgent(nn.Module):
    """Predict-before-feedback API with a persistent, caller-owned belief state.

    Scores are factual outcome logits, not identified causal effects or an
    already-qualified policy. No logged propensity or current reward is an
    input to predict(). Features must be available before the decision.
    """
    def __init__(self, context_size: int, action_size: int, outcome_size: int = 1):
        super().__init__()
        if min(context_size, action_size, outcome_size) < 1:
            raise ValueError("positive feature dimensions required")
        self.context_size, self.action_size = context_size, action_size
        self.outcome_size = outcome_size
        self.observation = nn.Linear(context_size, 16)
        self.enc = nn.Linear(16, 32)
        self.transition = nn.Linear(32, 32, bias=False)
        self.read_fast = nn.Linear(8, 32, bias=False)
        self.gate = nn.Linear(56, 32)
        self.slow = nn.Linear(72, 32)
        self.slow_gate = nn.Linear(64, 32)
        self.query = nn.Linear(64, 8)
        self.key = nn.Linear(48, 8)
        self.value = nn.Linear(64, 8)
        self.write = nn.Linear(64, 1)
        self.retention = nn.Parameter(torch.tensor(math.log(.04/.0099)))
        self.action = nn.Linear(action_size, 8)
        self.readout = nn.Sequential(nn.Linear(context_size + 64 + 16, 64),
                                     nn.SiLU(), nn.Linear(64, outcome_size))
        self.next_observation = nn.Linear(64 + 8, context_size)
        self.verify_map = nn.Linear(64 + 8 + outcome_size, 128)
        self.verify_key = nn.Linear(8, 8)
        self.verify_value = nn.Linear(outcome_size, 8)
        nn.init.orthogonal_(self.transition.weight, gain=.9)
        nn.init.normal_(self.slow.weight, std=.01)
        nn.init.zeros_(self.slow.bias)
        with torch.no_grad():
            self.gate.bias.copy_(-torch.log(torch.logspace(math.log10(2), math.log10(16), 32)-1))
            self.slow_gate.bias.copy_(-torch.log(torch.logspace(math.log10(8), math.log10(64), 32)-1))

    @staticmethod
    def address(x):
        return x / torch.sqrt(x.square().sum(-1, keepdim=True) + 1e-5)

    def initial(self, batch_size: int) -> Belief:
        if batch_size < 1:
            raise ValueError("positive batch size required")
        p = self.retention
        return Belief(p.new_zeros(batch_size, 32), p.new_zeros(batch_size, 32),
                      p.new_zeros(batch_size, 8, 8))

    def _validate(self, state):
        b = state.fast.shape[0]
        if state.fast.shape != (b,32) or state.slow.shape != (b,32) or state.memory.shape != (b,8,8):
            raise ValueError("expected fast32/slow32/matrix8x8")
        return b

    def observe(self, state: Belief, context: torch.Tensor) -> Belief:
        b = self._validate(state)
        if context.shape != (b, self.context_size):
            raise ValueError("context shape mismatch")
        u = torch.tanh(self.observation(context))
        f, s, matrix = state
        q = self.address(self.query(torch.cat((f,s), -1)))
        recalled = (matrix @ q.unsqueeze(-1)).squeeze(-1)
        gate = torch.sigmoid(self.gate(torch.cat((u,f,recalled), -1)))
        f = f + gate * (torch.tanh(self.enc(u)+self.transition(f)+self.read_fast(recalled))-f)
        joint = torch.cat((f,s), -1)
        key = self.address(self.key(torch.cat((u,f), -1)))
        value = torch.tanh(self.value(joint))
        beta = torch.sigmoid(self.write(joint))
        retained = (.95 + .0499 * torch.sigmoid(self.retention)) * matrix
        error = value - (retained @ key.unsqueeze(-1)).squeeze(-1)
        matrix = retained + beta.unsqueeze(-1) * error.unsqueeze(-1) * key.unsqueeze(-2)
        recalled = (matrix @ q.unsqueeze(-1)).squeeze(-1)
        innovation = torch.tanh(self.slow(torch.cat((f-s,s,recalled), -1)))
        s = s + torch.sigmoid(self.slow_gate(joint)) * (torch.tanh(s+innovation)-s)
        return Belief(f, s, matrix)

    def predict(self, state: Belief, context: torch.Tensor, candidates: torch.Tensor):
        """B x A x action_size -> logits B x A x outcome_size and next observation.

        Querying any number of candidate actions never writes state. The next
        observation head requires genuinely linked transitions for training.
        """
        b = self._validate(state)
        if context.shape != (b,self.context_size) or candidates.ndim != 3 or candidates.shape[0] != b or candidates.shape[2] != self.action_size:
            raise ValueError("context/action shape mismatch")
        a = torch.tanh(self.action(candidates))
        joint = torch.cat((state.fast,state.slow), -1)
        latent = joint.unsqueeze(1).expand(-1,a.shape[1],-1)
        recall = (state.memory.unsqueeze(1) @ self.address(a).unsqueeze(-1)).squeeze(-1)
        x = torch.cat((context.unsqueeze(1).expand(-1,a.shape[1],-1),latent,a,recall), -1)
        return self.readout(x), self.next_observation(torch.cat((latent,a), -1))

    def verify(self, state: Belief, executed_action: torch.Tensor,
               outcome: torch.Tensor, predicted_outcome: torch.Tensor,
               available: torch.Tensor) -> Belief:
        """Assimilate only actually arrived outcomes, after the logged action.

        Caller supplies prediction in the SAME outcome units (e.g. sigmoid of
        binary logits), made before feedback. Missing labels may be NaN; they
        are removed before arithmetic and cause exactly zero state change.
        The fast/slow correction has coordinate magnitude <= .25 per event.
        This is not a guarantee against forgetting or a global stability proof.
        """
        b = self._validate(state)
        if executed_action.shape != (b,self.action_size) or outcome.shape != (b,self.outcome_size) or predicted_outcome.shape != outcome.shape or available.shape != (b,1):
            raise ValueError("verification shape mismatch")
        if available.dtype != torch.bool:
            raise ValueError("availability must be a boolean observed-outcome mask")
        if not (torch.isfinite(outcome[available.expand_as(outcome)]).all() and
                torch.isfinite(predicted_outcome[available.expand_as(outcome)]).all()):
            raise ValueError("available outcome and preceding prediction must be finite")
        # Missing rows never enter residual arithmetic, including NaN predictions.
        residual = torch.where(available, outcome, torch.zeros_like(outcome)) - torch.where(available, predicted_outcome.detach(), torch.zeros_like(outcome))
        residual = torch.tanh(residual)
        action = torch.tanh(self.action(executed_action))
        joint = torch.cat((state.fast,state.slow), -1)
        proposal, gate = self.verify_map(torch.cat((joint,action,residual), -1)).chunk(2,-1)
        delta = .25 * torch.sigmoid(gate) * torch.tanh(proposal)
        corrected = joint + torch.where(available,delta,torch.zeros_like(delta))
        key = self.address(self.verify_key(action))
        value = torch.tanh(self.verify_value(residual))
        error = value - (state.memory @ key.unsqueeze(-1)).squeeze(-1)
        update = .25 * error.unsqueeze(-1) * key.unsqueeze(-2)
        matrix = state.memory + torch.where(available.unsqueeze(-1),update,torch.zeros_like(update))
        return Belief(corrected[:,:32], corrected[:,32:], matrix)
