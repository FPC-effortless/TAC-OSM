"""ES-WAM prototype: exchange-structured memory and nonauthoritative world-action forecasts.

No quantum-mechanical claim. No real-world trained capability is implied.
Memory identity is supplied by the source; model never invents entity identity.
Only a preregistered live-guarded experiment may fit the complete neural model.
"""
from __future__ import annotations

from typing import NamedTuple
import torch
from torch import nn
from torch.nn import functional as F


class ExchangeState(NamedTuple):
    ids: torch.Tensor              # B,S; -1 means free
    keys: torch.Tensor             # B,S,D
    values: torch.Tensor           # B,S,R,D
    occupied: torch.Tensor         # B,S,R


class WorldState(NamedTuple):
    fast: torch.Tensor
    slow: torch.Tensor
    memory: ExchangeState


class WorldActionForecast(NamedTuple):
    mean_next: torch.Tensor
    scale_next: torch.Tensor
    proposed_action: torch.Tensor
    state: WorldState
    previous_prediction_present: torch.Tensor
    observation_written: torch.Tensor
    prediction_written: torch.Tensor


class ExchangeMemory(nn.Module):
    """Finite entity-addressed storage; observable collision refusal, never silent eviction.

    Roles: 0=observed, 1=unverified prediction, 2=verified outcome.
    'verified' is a caller-authorized factual designation, not inferred by the model.
    """
    roles = 3

    def __init__(self, width: int = 16, slots: int = 8):
        super().__init__()
        if width < 1 or slots < 1:
            raise ValueError("positive capacity required")
        self.width, self.slots = width, slots
        self.exchange_map = nn.Linear(2 * width, width)
        self.merge_gate = nn.Linear(2 * width, 1)

    def initial(self, batch: int, *, device=None, dtype=None) -> ExchangeState:
        if batch < 1:
            raise ValueError("batch must be positive")
        dtype = dtype or torch.float32
        return ExchangeState(
            torch.full((batch, self.slots), -1, device=device, dtype=torch.long),
            torch.zeros(batch, self.slots, self.width, device=device, dtype=dtype),
            torch.zeros(batch, self.slots, self.roles, self.width, device=device, dtype=dtype),
            torch.zeros(batch, self.slots, self.roles, device=device, dtype=torch.bool),
        )

    def exchange(self, a: torch.Tensor, b: torch.Tensor):
        if a.shape != b.shape or a.shape[-1] != self.width:
            raise ValueError("paired representation dimension mismatch")
        forward = self.exchange_map(torch.cat((a, b), -1))
        backward = self.exchange_map(torch.cat((b, a), -1))
        return (forward + backward) * 0.5, (forward - backward) * 0.5

    def write(self, state: ExchangeState, ids: torch.Tensor, role: int,
              keys: torch.Tensor, values: torch.Tensor,
              enabled: torch.Tensor) -> tuple[ExchangeState, torch.Tensor]:
        b, d = values.shape
        if role not in range(self.roles):
            raise ValueError("unknown evidence role")
        if (ids.shape != (b,) or ids.dtype != torch.long or (ids < 0).any()
                or keys.shape != (b, d) or d != self.width
                or enabled.shape != (b,) or enabled.dtype != torch.bool
                or state.ids.shape != (b, self.slots)
                or state.values.shape != (b, self.slots, self.roles, d)
                or state.keys.shape != (b, self.slots, d)
                or state.occupied.shape != (b, self.slots, self.roles)):
            raise ValueError("invalid memory state, identity or write shape")
        if not (torch.isfinite(keys).all() and torch.isfinite(values).all()):
            raise ValueError("nonfinite memory write")
        new_ids, new_keys = state.ids.clone(), state.keys.clone()
        new_values, new_occupied = state.values.clone(), state.occupied.clone()
        admitted = torch.zeros(b, device=ids.device, dtype=torch.bool)
        for i in range(b):
            if not bool(enabled[i]):
                continue
            match = (new_ids[i] == ids[i]).nonzero(as_tuple=True)[0]
            if len(match):
                slot = int(match[0])
            else:
                free = (new_ids[i] == -1).nonzero(as_tuple=True)[0]
                if not len(free):
                    continue  # fail closed; no undisclosed replacement
                slot = int(free[0])
                new_ids[i, slot] = ids[i]
                new_keys[i, slot] = F.normalize(keys[i], dim=0, eps=1e-6)
            if bool(new_occupied[i, slot, role]):
                old = new_values[i, slot, role].clone()
                incoming = values[i]
                symmetric, antisymmetric = self.exchange(old, incoming)
                # Learned merge rate; antisymmetry gives direction-sensitive
                # information but is NOT an automatic physical exclusion rule.
                mix = self.merge_gate(torch.cat((old + symmetric, incoming + antisymmetric), -1))
                updated = torch.lerp(old, incoming, mix.sigmoid())
            else:
                updated = values[i]
            new_values[i, slot, role] = updated
            new_occupied[i, slot, role] = True
            admitted[i] = True
        return ExchangeState(new_ids, new_keys, new_values, new_occupied), admitted

    def retrieve(self, state: ExchangeState, query: torch.Tensor, role: int,
                 requested_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if role not in range(self.roles) or query.ndim != 2 or query.shape[-1] != self.width:
            raise ValueError("invalid query shape or evidence role")
        b = len(query)
        if state.ids.shape != (b, self.slots):
            raise ValueError("batch/address mismatch")
        if requested_ids is not None and (requested_ids.shape != (b,) or requested_ids.dtype != torch.long):
            raise ValueError("requested identities must be observed integer IDs")
        results, hits = [], []
        for i in range(b):
            valid = state.occupied[i, :, role] & (state.ids[i] >= 0)
            if requested_ids is not None:
                valid = valid & (state.ids[i] == requested_ids[i])
            if not bool(valid.any()):
                results.append(query[i].new_zeros(self.width))
                hits.append(False)
                continue
            sims = state.keys[i, valid] @ F.normalize(query[i], dim=0, eps=1e-6)
            weights = torch.softmax(sims, dim=0)
            results.append((weights[:, None] * state.values[i, valid, role]).sum(0))
            hits.append(True)
        return torch.stack(results), torch.tensor(hits, dtype=torch.bool, device=query.device)


class ExchangeWorldActionModel(nn.Module):
    """One native fast/slow world model with protected observed/predicted roles.

    This produces only forecasts and nonauthoritative action *proposals*.
    It does not learn an action policy or causal outcomes without suitable data.
    """
    architecture = "eswam_exchange_world_action_candidate"

    def __init__(self, dimension: int = 14, action_dim: int = 14,
                 width: int = 90, memory_width: int = 16, slots: int = 8,
                 state_scale=None):
        super().__init__()
        if dimension < 1 or action_dim < 1:
            raise ValueError("source dimensions must be positive")
        self.dimension, self.action_dim, self.width = dimension, action_dim, width
        self.memory = ExchangeMemory(memory_width, slots)
        self.encoder = nn.Sequential(
            nn.Linear(2 * dimension + action_dim + 1, width), nn.SiLU(),
            nn.Linear(width, width), nn.Tanh())
        self.fast_map = nn.Linear(2 * width, 2 * width)
        self.slow_map = nn.Linear(2 * width, 2 * width)
        self.key_map = nn.Linear(width, memory_width)
        self.value_map = nn.Linear(width, memory_width)
        self.query_map = nn.Linear(width, memory_width)
        self.verification_map = nn.Linear(memory_width, width)
        self.readout = nn.Linear(2 * width + memory_width, 64)
        self.next_heads = nn.Linear(64, 2 * dimension)
        self.action_head = nn.Linear(64 + action_dim, action_dim)
        self.predicted_code = nn.Linear(64, memory_width)
        scale = torch.ones(dimension) if state_scale is None else torch.as_tensor(state_scale, dtype=torch.float32)
        if scale.shape != (dimension,) or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("finite positive physical source scales required")
        self.register_buffer("state_scale", scale.clone().detach())

    def initial(self, batch: int, *, device=None, dtype=None) -> WorldState:
        dtype = dtype or self.state_scale.dtype
        return WorldState(torch.zeros(batch, self.width, device=device, dtype=dtype),
                          torch.zeros(batch, self.width, device=device, dtype=dtype),
                          self.memory.initial(batch, device=device, dtype=dtype))

    def forward(self, observation: torch.Tensor, action_at_t: torch.Tensor,
                elapsed_seconds: torch.Tensor, entity_ids: torch.Tensor,
                observed_mask: torch.Tensor, goal: torch.Tensor | None = None,
                state: WorldState | None = None) -> WorldActionForecast:
        if (observation.ndim != 2 or observation.shape[-1] != self.dimension
                or action_at_t.shape != (len(observation), self.action_dim)
                or observed_mask.shape != observation.shape or observed_mask.dtype != torch.bool
                or elapsed_seconds.shape != (len(observation),)
                or entity_ids.shape != (len(observation),) or entity_ids.dtype != torch.long
                or (entity_ids < 0).any()):
            raise ValueError("factual observation/action/identity contract violated")
        if goal is None:
            goal = action_at_t.new_zeros(len(observation), self.action_dim)
        if goal.shape != action_at_t.shape:
            raise ValueError("goal dimension mismatch")
        if not (torch.isfinite(observation[observed_mask]).all()
                and torch.isfinite(action_at_t).all() and torch.isfinite(goal).all()
                and torch.isfinite(elapsed_seconds).all()
                and (elapsed_seconds >= 0).all()):
            raise ValueError("nonfinite known data or invalid elapsed time")
        if state is None:
            state = self.initial(len(observation), device=observation.device, dtype=observation.dtype)
        if state.fast.shape != (len(observation), self.width) or state.slow.shape != state.fast.shape:
            raise ValueError("invalid recurrent state")
        current = torch.where(observed_mask, observation, torch.zeros_like(observation))
        x = torch.cat((current, observed_mask.to(current.dtype), action_at_t,
                       torch.log1p(elapsed_seconds).unsqueeze(-1)), -1)
        z = self.encoder(x)
        key = F.normalize(self.key_map(z), dim=-1, eps=1e-6)
        old_prediction, has_prediction = self.memory.retrieve(
            state.memory, key, 1, requested_ids=entity_ids)
        observed_code = torch.tanh(self.value_map(z))
        innovation = torch.where(has_prediction[:, None],
                                 observed_code - old_prediction, torch.zeros_like(observed_code))
        corrected_fast = state.fast + 0.05 * torch.tanh(self.verification_map(innovation))
        candidate, gate = self.fast_map(torch.cat((z, corrected_fast), -1)).chunk(2, -1)
        fast = torch.lerp(corrected_fast.tanh(), candidate.tanh(), gate.sigmoid()).tanh()
        candidate_s, gate_s = self.slow_map(torch.cat((fast, state.slow), -1)).chunk(2, -1)
        slow = torch.lerp(state.slow, candidate_s.tanh(), (0.25 * gate_s.sigmoid())).tanh()
        # Missingness does not authorize treating all-missing as a factual WRITE.
        write_enabled = observed_mask.any(-1)
        memory, written = self.memory.write(state.memory, entity_ids, 0, key,
                                            observed_code, write_enabled)
        remembered, _ = self.memory.retrieve(memory, self.query_map(z), 0)
        decoded = F.silu(self.readout(torch.cat((fast, slow, remembered), -1)))
        heads = self.next_heads(decoded)
        d = self.dimension
        mean_next = current + self.state_scale * (0.1 * heads[:, :d].tanh())
        scale_next = self.state_scale * (F.softplus(heads[:, d:]) + 1e-4)
        proposed_action = torch.tanh(self.action_head(torch.cat((decoded, goal), -1)))
        # Predictions occupy a distinct unverified role; never overwrite facts.
        memory, predicted_written = self.memory.write(
            memory, entity_ids, 1, key, self.predicted_code(decoded).tanh(),
            torch.ones(len(observation), device=observation.device, dtype=torch.bool))
        return WorldActionForecast(mean_next, scale_next, proposed_action,
                                   WorldState(fast, slow, memory), has_prediction,
                                   written, predicted_written)
