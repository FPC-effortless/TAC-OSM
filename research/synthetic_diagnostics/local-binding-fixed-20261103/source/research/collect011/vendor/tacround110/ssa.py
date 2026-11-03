"""A new SSA-inspired causal slot-memory adapter, not an SSA008 reproduction.

Primary SSA008 implementation/results were not recovered; see
``SSA_SOURCE_REVIEW.md``. In particular the usage-based survival proxy below is
NOT TAC-Prime's six-signal NSF scorer or a validated structure-discovery model.
All new equations and omission semantics are explicit in this module.

For observation h_t, normalize query q_t and previous slot keys K. Rank
q_t.K_s / temperature + log(u_s+epsilon) (the last term exists only with NSF).
Stable descending sort breaks ties by lower slot index. Read the selected
PRIOR values with a softmax, then emit W_out(read). Current observations are
written only AFTER that read, and no future target enters this API.

With Hebbian writes, eta=sigmoid(W_admit h)*sigmoid(W_plastic h), sparse
p_s are the selected read weights, and r_s=eta*p_s. The new observed key is
normalize(W_key h), and value is tanh(W_value h). Retain toward learned priors
using alpha_s, then perform a convex coactivation-weighted key/value update:

  K_base=normalize(alpha*K_prev+(1-alpha)*K_prior)
  V_base=alpha*V_prev+(1-alpha)*V_prior
  K_new=normalize((1-r)*K_base+r*observed_key)
  V_new=(1-r)*V_base+r*observed_value.

NSF-inspired alpha=rho0+(1-rho0)*sigmoid(W_survH h + W_survK K_prev),
and survival trace u_new=alpha*u_prev+(1-alpha)*p. Without NSF, alpha=rho0
is fixed and survival heads/state are physically absent. Without Hebbian,
observation key/value/admission/plasticity projections are physically absent
and K/V stay unchanged; learned priors preserve an active direct slot path.
Every write-mask-false observation freezes ALL memory fields exactly.

Queries/projections are batched over T, but mutable-key/survival routing makes
the state trajectory TOKEN SERIAL. Ranking is dense O(S*K) plus stable sorting;
only selected values are gathered for the O(topk*V) read. Retention/state writes
still touch bounded dense slot arrays. No sparse-search or speed gain is claimed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SSAState:
    owners: tuple[str, ...]
    clock: int
    keys: Tensor
    values: Tensor
    survival: Tensor | None
    config_signature: str

    def detach(self):
        return SSAState(self.owners, self.clock, self.keys.detach(), self.values.detach(),
                        None if self.survival is None else self.survival.detach(),
                        self.config_signature)

    def state_bytes(self):
        values = [self.keys, self.values]
        if self.survival is not None:
            values.append(self.survival)
        return sum(x.numel() * x.element_size() for x in values)

    tensor_bytes = state_bytes

    def payload(self):
        return {"schema": "tacround110.ssa-inspired-state.v1", "owners": self.owners,
                "clock": self.clock, "keys": self.keys, "values": self.values,
                "survival": self.survival, "config_signature": self.config_signature}

    @classmethod
    def from_payload(cls, payload):
        expected = {"schema", "owners", "clock", "keys", "values", "survival", "config_signature"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("complete SSA state payload required")
        if payload["schema"] != "tacround110.ssa-inspired-state.v1":
            raise ValueError("unsupported SSA state schema")
        if not isinstance(payload["owners"], (tuple, list)):
            raise ValueError("ordered SSA owners required")
        if not isinstance(payload["keys"], Tensor) or not isinstance(payload["values"], Tensor):
            raise ValueError("SSA keys and values must be tensors")
        if payload["survival"] is not None and not isinstance(payload["survival"], Tensor):
            raise ValueError("SSA survival must be a tensor or None")
        # Config-dependent shapes/device/precision are checked by the model.
        return cls(tuple(payload["owners"]), payload["clock"], payload["keys"].clone(),
                   payload["values"].clone(),
                   None if payload["survival"] is None else payload["survival"].clone(),
                   payload["config_signature"])


class SparseStructureMemory(nn.Module):
    """Untrained slot-memory adaptation with explicit Hebbian/NSF omission arms."""

    def __init__(self, d_model, slots=32, key_dim=16, value_dim=16, topk=4,
                 hebbian=True, nsf=True, *, fixed_retention=.95,
                 temperature=1., epsilon=1e-6):
        super().__init__()
        for name, value in (("d_model", d_model), ("slots", slots), ("key_dim", key_dim),
                            ("value_dim", value_dim), ("topk", topk)):
            if type(value) is not int or value < 1:
                raise ValueError(name + " must be a positive integer")
        if topk > slots or type(hebbian) is not bool or type(nsf) is not bool:
            raise ValueError("topk<=slots and boolean Hebbian/NSF flags required")
        if (not math.isfinite(float(fixed_retention)) or not 0 <= fixed_retention < 1
                or not math.isfinite(float(temperature)) or temperature <= 0
                or not math.isfinite(float(epsilon)) or epsilon <= 0):
            raise ValueError("legal finite retention, temperature and epsilon required")
        self.d_model, self.slots, self.key_dim, self.value_dim, self.topk = (
            d_model, slots, key_dim, value_dim, topk)
        self.hebbian, self.nsf = hebbian, nsf
        self.fixed_retention = float(fixed_retention)
        self.temperature, self.epsilon = float(temperature), float(epsilon)
        self.query = nn.Linear(d_model, key_dim, bias=False)
        self.readout = nn.Linear(value_dim, d_model, bias=False)
        # These learned priors are counted model capacity, not hidden mutable state.
        self.initial_keys = nn.Parameter(torch.randn(slots, key_dim) / math.sqrt(key_dim))
        self.initial_values = nn.Parameter(.02 * torch.randn(slots, value_dim))
        if hebbian:
            self.key = nn.Linear(d_model, key_dim, bias=False)
            self.value = nn.Linear(d_model, value_dim, bias=False)
            self.admission = nn.Linear(d_model, 1)
            self.plasticity = nn.Linear(d_model, 1)
        if nsf:
            self.survival_query = nn.Linear(d_model, 1)
            self.survival_key = nn.Linear(key_dim, 1, bias=False)
        declaration = {"family": "SSA-inspired-new-causal-adapter-v1", "d_model": d_model,
            "slots": slots, "key_dim": key_dim, "value_dim": value_dim, "topk": topk,
            "hebbian": hebbian, "nsf": nsf, "fixed_retention": self.fixed_retention,
            "temperature": self.temperature, "epsilon": self.epsilon}
        self.config_signature = hashlib.sha256(json.dumps(declaration, sort_keys=True,
                                                        separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def _owners(owners):
        if (not isinstance(owners, (tuple, list)) or not owners
                or any(type(x) is not str or not x for x in owners)
                or len(set(owners)) != len(owners)):
            raise ValueError("unique ordered nonempty SSA owner strings required")
        return tuple(owners)

    def _priors(self):
        return (F.normalize(self.initial_keys, dim=-1, eps=self.epsilon),
                self.initial_values.tanh())

    def start(self, owners):
        owners = self._owners(owners)
        key, value = self._priors()
        count = len(owners)
        return SSAState(owners, 0, key.unsqueeze(0).expand(count, -1, -1).clone(),
            value.unsqueeze(0).expand(count, -1, -1).clone(),
            key.new_full((count, self.slots), .5) if self.nsf else None,
            self.config_signature)

    def validate(self, h, state, owners):
        owners = self._owners(owners)
        weight = self.initial_keys
        if (not isinstance(h, Tensor) or h.ndim != 3 or min(h.shape) < 1
                or h.shape[-1] != self.d_model or h.dtype not in (torch.float32, torch.float64)
                or h.device != weight.device or h.dtype != weight.dtype):
            raise ValueError("nonempty same-device/precision Float[B,T,d_model] required")
        if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
            raise ValueError("SSA mixed precision is not qualified")
        if (not isinstance(state, SSAState) or state.owners != owners
                or type(state.clock) is not int or state.clock < 0
                or len(owners) != h.shape[0] or state.config_signature != self.config_signature):
            raise ValueError("same-owner complete SSA state/configuration required")
        geometry = ((state.keys, (h.shape[0], self.slots, self.key_dim)),
                    (state.values, (h.shape[0], self.slots, self.value_dim)))
        if self.nsf:
            geometry += ((state.survival, (h.shape[0], self.slots)),)
        elif state.survival is not None:
            raise ValueError("no-NSF state cannot contain survival dynamics")
        flags = [torch.isfinite(h).all()]
        for value, shape in geometry:
            if (not isinstance(value, Tensor) or value.shape != shape
                    or value.dtype != weight.dtype or value.device != weight.device):
                raise ValueError("SSA state shape/device/precision mismatch")
            flags.append(torch.isfinite(value).all())
        if self.nsf:
            flags.append(((state.survival >= 0) & (state.survival <= 1)).all())
        if not bool(torch.stack(flags).all()):
            raise ValueError("finite SSA observations/state and bounded survival required")

    def forward(self, h, state=None, write_mask=None, *, owners=None):
        if owners is None:
            if state is not None:
                raise ValueError("explicit owners required when continuing SSA state")
            if not isinstance(h, Tensor) or h.ndim != 3:
                raise ValueError("Float[B,T,D] SSA observations required")
            owners = tuple(f"ssa-fresh-{i}" for i in range(h.shape[0]))
        owners = self._owners(owners)
        if state is None:
            state = self.start(owners)
        self.validate(h, state, owners)
        if write_mask is None:
            write_mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        if (not isinstance(write_mask, Tensor) or write_mask.dtype != torch.bool
                or write_mask.shape != h.shape[:2] or write_mask.device != h.device):
            raise ValueError("SSA write_mask must be Bool[B,T] on observation device")
        queries = F.normalize(self.query(h), dim=-1, eps=self.epsilon)
        if self.hebbian:
            write_keys = F.normalize(self.key(h), dim=-1, eps=self.epsilon)
            write_values = self.value(h).tanh()
            rates = (self.admission(h).sigmoid() * self.plasticity(h).sigmoid()).squeeze(-1)
        if self.nsf:
            survival_queries = self.survival_query(h)
        prior_key, prior_value = self._priors()
        keys, values, survival = state.keys, state.values, state.survival
        reads = []
        for time in range(h.shape[1]):
            addresses = F.normalize(keys, dim=-1, eps=self.epsilon)
            scores = (addresses * queries[:, time, None, :]).sum(-1) / self.temperature
            if self.nsf:
                scores = scores + (survival + self.epsilon).log()
            # Full dense score/sort remains explicit; only value-read work is gathered.
            selected = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :self.topk]
            selected_scores = scores.gather(1, selected)
            probabilities = selected_scores.softmax(-1)
            selected_values = values.gather(1, selected[..., None].expand(-1, -1, self.value_dim))
            reads.append((probabilities[..., None] * selected_values).sum(1))
            sparse = scores.new_zeros(scores.shape).scatter(1, selected, probabilities)
            eligible = write_mask[:, time]
            if self.nsf:
                alpha = self.fixed_retention + (1 - self.fixed_retention) * (
                    survival_queries[:, time, None, :] + self.survival_key(keys)).sigmoid().squeeze(-1)
                candidate_survival = alpha * survival + (1 - alpha) * sparse
            else:
                alpha = self.fixed_retention
            if self.hebbian:
                rate = rates[:, time, None] * sparse
                keep = alpha[..., None] if isinstance(alpha, Tensor) else alpha
                base_key = F.normalize(keep * keys + (1 - keep) * prior_key,
                                       dim=-1, eps=self.epsilon)
                base_value = keep * values + (1 - keep) * prior_value
                candidate_key = F.normalize((1 - rate[..., None]) * base_key +
                    rate[..., None] * write_keys[:, time, None, :], dim=-1, eps=self.epsilon)
                candidate_value = ((1 - rate[..., None]) * base_value +
                                   rate[..., None] * write_values[:, time, None, :])
                keys = torch.where(eligible[:, None, None], candidate_key, keys)
                values = torch.where(eligible[:, None, None], candidate_value, values)
            if self.nsf:
                survival = torch.where(eligible[:, None], candidate_survival, survival)
        residual = self.readout(torch.stack(reads, dim=1))
        final = SSAState(owners, state.clock + h.shape[1],
            keys.clone(memory_format=torch.contiguous_format),
            values.clone(memory_format=torch.contiguous_format),
            None if survival is None else survival.clone(memory_format=torch.contiguous_format),
            self.config_signature)
        return residual, final

    def step(self, h, *, state, owners, write_mask=None):
        if not isinstance(h, Tensor) or h.ndim != 3 or h.shape[1] != 1:
            raise ValueError("SSA step requires Float[B,1,D]")
        return self(h, state=state, owners=owners, write_mask=write_mask)
