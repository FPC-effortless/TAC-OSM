"""LATFUSE001: UNTRAINED proposed multi-memory operators, not a qualified model.

Only the affine temporal subgraph admits associative parallel prefix composition.
All inputs must be authorized at prediction time by an external source adapter.
No attempt to infer causal interventions from logged observations is made.
"""
from typing import NamedTuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FusionState(NamedTuple):
    fast: Tensor          # [batch, hidden]
    slow: Tensor          # [batch, hidden]
    associative: Tensor   # [batch, values, keys]
    entities: Tensor      # [batch, entity_count, hidden]
    active: Tensor        # [batch, entity_count], caller-authorized occupancy


def affine_compose(first: tuple[Tensor, Tensor], second: tuple[Tensor, Tensor]):
    """Apply first, then second: second(first(z)). Broadcastable diagonal affine pairs."""
    a, b = first
    c, d = second
    return c * a, c * b + d


def affine_prefix(a: Tensor, b: Tensor, initial: Tensor) -> Tensor:
    """Reference inclusive scan, O(T log T) ops; NOT an optimized GPU scan.

    Shapes [T,B,D], [T,B,D], and [B,D]. No recurrent nonlinear operations
    inside this associative prefix; training performance is unmeasured.
    """
    if a.shape != b.shape or a.ndim != 3 or initial.shape != a.shape[1:]:
        raise ValueError("affine scan shape mismatch")
    if a.shape[0] == 0:
        return a
    aa, bb = a, b
    stride = 1
    while stride < a.shape[0]:
        # Functional/differentiable Hillis-Steele prefix with an immutable prior level.
        part_a, part_b = affine_compose((aa[:-stride], bb[:-stride]), (aa[stride:], bb[stride:]))
        aa = torch.cat((aa[:stride], part_a), dim=0)
        bb = torch.cat((bb[:stride], part_b), dim=0)
        stride *= 2
    return aa * initial.unsqueeze(0) + bb


def gated_delta(matrix: Tensor, key: Tensor, value: Tensor,
                retention: Tensor, write: Tensor) -> Tensor:
    """Rectangular gated delta matrix edit with one normalized key per batch.

    The retention gate affects all columns. Source-entity isolation is provided
    by the separate entity bank, NOT by this associative matrix.
    """
    if matrix.ndim != 3 or key.shape != (matrix.shape[0], matrix.shape[2]) or value.shape != (matrix.shape[0], matrix.shape[1]):
        raise ValueError("key/value/matrix shape mismatch")
    if retention.shape != (matrix.shape[0], 1) or write.shape != (matrix.shape[0], 1):
        raise ValueError("gate shape mismatch")
    if not (torch.isfinite(matrix).all() and torch.isfinite(key).all() and torch.isfinite(value).all()
            and torch.isfinite(retention).all() and torch.isfinite(write).all()):
        raise ValueError("nonfinite delta input")
    if bool(((retention < 0) | (retention > 1) | (write < 0) | (write > 1)).any()):
        raise ValueError("retention and write gates must lie in [0,1]")
    k = F.normalize(key, dim=-1, eps=1e-8)
    retained = retention.unsqueeze(-1) * matrix
    prediction = (retained @ k.unsqueeze(-1)).squeeze(-1)
    return retained + write.unsqueeze(-1) * (value - prediction).unsqueeze(-1) * k.unsqueeze(-2)


def source_slot_write(slots: Tensor, active: Tensor, slot_id: Tensor,
                      new_value: Tensor, admission: Tensor):
    """Update one explicitly source-authorized slot; never hash entity IDs.

    Invalid IDs must fail before any indexing; admission [B,1] in [0,1].
    Occupancy is boolean and becomes true only when an actual write occurs.
    """
    if slots.ndim != 3 or active.shape != slots.shape[:2] or active.dtype != torch.bool:
        raise ValueError("entity state shape/type mismatch")
    b, count, dim = slots.shape
    if slot_id.shape != (b,) or slot_id.dtype not in (torch.int64, torch.int32) or new_value.shape != (b, dim) or admission.shape != (b, 1):
        raise ValueError("slot write shape/type mismatch")
    if bool(((slot_id < 0) | (slot_id >= count)).any()):
        raise ValueError("source slot outside registered capacity")
    if not (torch.isfinite(new_value).all() and torch.isfinite(admission).all()):
        raise ValueError("nonfinite slot write")
    if bool(((admission < 0) | (admission > 1)).any()):
        raise ValueError("admission outside [0,1]")
    selection = F.one_hot(slot_id.to(torch.long), count).to(slots.dtype).unsqueeze(-1)
    weight = selection * admission.unsqueeze(-1)
    updated = slots + weight * (new_value.unsqueeze(1) - slots)
    becomes_active = selection.squeeze(-1).bool() & (admission > 0)
    return updated, active | becomes_active


class FusedLatentCandidate(nn.Module):
    """A small no-fit integrated graph. NOT 100k-budget qualified for training.

    Keeps source-authenticated slots separate from content-addressed fast weights;
    affine temporal states can be parallelized if all embeddings are prepared.
    Forecasts condition on provided candidates, not identified do()-effects.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 32,
                 key_dim: int = 16, val_dim: int = 16, entity_count: int = 16):
        super().__init__()
        if min(obs_dim, action_dim, hidden, key_dim, val_dim, entity_count) < 1:
            raise ValueError("dimensions must be positive")
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.hidden, self.key_dim, self.val_dim, self.entity_count = hidden, key_dim, val_dim, entity_count
        self.center = nn.Parameter(torch.zeros(obs_dim))
        self.log_decay = nn.Parameter(torch.zeros(obs_dim))
        self.embed = nn.Sequential(nn.Linear(obs_dim * 3, hidden * 2), nn.GELU(),
                                   nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden))
        self.temporal_a = nn.Linear(hidden, hidden)
        self.temporal_b = nn.Linear(hidden, hidden)
        self.slow_input = nn.Linear(hidden, hidden)
        self.key_map = nn.Linear(hidden, key_dim)
        self.val_map = nn.Linear(hidden, val_dim)
        self.retention_map = nn.Linear(hidden, 1)
        self.write_map = nn.Linear(hidden, 1)
        self.entity_map = nn.Linear(hidden, hidden)
        self.entity_gate = nn.Linear(hidden, 1)
        self.action_map = nn.Linear(action_dim, key_dim)
        self.forecast_head = nn.Sequential(nn.Linear(hidden * 3 + val_dim + key_dim, hidden * 2),
                                           nn.GELU(), nn.Linear(hidden * 2, obs_dim * 2))

    def initial(self, batch: int, *, device=None, dtype=None) -> FusionState:
        if batch < 1:
            raise ValueError("batch must be positive")
        kw = dict(device=self.center.device if device is None else device,
                  dtype=self.center.dtype if dtype is None else dtype)
        return FusionState(torch.zeros(batch,self.hidden,**kw), torch.zeros(batch,self.hidden,**kw),
                           torch.zeros(batch,self.val_dim,self.key_dim,**kw),
                           torch.zeros(batch,self.entity_count,self.hidden,**kw),
                           torch.zeros(batch,self.entity_count,device=kw['device'],dtype=torch.bool))

    def encode(self, x: Tensor, observed: Tensor, elapsed: Tensor, last_seen: Tensor):
        b = x.shape[0]
        if x.shape != (b,self.obs_dim) or observed.shape != x.shape or observed.dtype != torch.bool or elapsed.shape != x.shape or last_seen.shape != x.shape:
            raise ValueError("observation/mask/time shape mismatch")
        if not (torch.isfinite(elapsed).all() and bool((elapsed >= 0).all())):
            raise ValueError("elapsed time must be finite and nonnegative")
        if not (torch.isfinite(x[observed]).all() and torch.isfinite(last_seen[~observed]).all()):
            raise ValueError("nonfinite source-observable input")
        # where-before-arithmetic prevents NaN labels in missing channels leaking into hidden state.
        recent = torch.where(observed, x, torch.zeros_like(x))
        former = torch.where(observed, torch.zeros_like(last_seen), last_seen)
        decay = torch.exp(-F.softplus(self.log_decay).unsqueeze(0) * elapsed)
        imputed = torch.where(observed, recent,
                              decay * former + (1 - decay) * self.center.unsqueeze(0))
        features = torch.cat((imputed, observed.to(imputed.dtype), torch.log1p(elapsed)), dim=-1)
        return self.embed(features)

    def step(self, state: FusionState, x: Tensor, observed: Tensor, elapsed: Tensor,
             last_seen: Tensor, source_slot: Tensor) -> FusionState:
        b = x.shape[0]
        if state.fast.shape != (b,self.hidden) or state.slow.shape != (b,self.hidden) or state.associative.shape != (b,self.val_dim,self.key_dim) or state.entities.shape != (b,self.entity_count,self.hidden) or state.active.shape != (b,self.entity_count):
            raise ValueError("state shape mismatch")
        u = self.encode(x,observed,elapsed,last_seen)
        # Signed diagonal affine coefficients: input-dependent but not state-dependent.
        a = torch.tanh(self.temporal_a(u))
        z = a * state.fast + torch.tanh(self.temporal_b(u))
        s = .99 * state.slow + .01 * torch.tanh(self.slow_input(u))
        key = self.key_map(u)
        memory = gated_delta(state.associative,key,torch.tanh(self.val_map(u)),
                             torch.sigmoid(self.retention_map(u)),torch.sigmoid(self.write_map(u)))
        entities,active = source_slot_write(state.entities,state.active,source_slot,
                                            torch.tanh(self.entity_map(u)),torch.sigmoid(self.entity_gate(u)))
        return FusionState(z,s,memory,entities,active)

    def forecast(self, state: FusionState, source_slot: Tensor, candidates: Tensor):
        """READ-ONLY conditional forecast [B,C,D]; mean and positive scale.

        Not a causal counterfactual and not automatically calibrated.
        """
        b = state.fast.shape[0]
        if candidates.ndim != 3 or candidates.shape[0] != b or candidates.shape[2] != self.action_dim or source_slot.shape != (b,):
            raise ValueError("forecast input shape mismatch")
        if bool(((source_slot < 0) | (source_slot >= self.entity_count)).any()):
            raise ValueError("invalid source slot")
        if not torch.isfinite(candidates).all():
            raise ValueError("nonfinite candidate")
        q = F.normalize(self.action_map(candidates),dim=-1,eps=1e-8)
        r = (state.associative.unsqueeze(1) @ q.unsqueeze(-1)).squeeze(-1)
        ix = source_slot.long().view(b,1,1).expand(-1,1,self.hidden)
        e = state.entities.gather(1,ix).squeeze(1)
        e = e * state.active.gather(1,source_slot.long().view(b,1)).to(e.dtype)
        n = candidates.shape[1]
        feature = torch.cat((state.fast[:,None,:].expand(-1,n,-1),
                             state.slow[:,None,:].expand(-1,n,-1),
                             e[:,None,:].expand(-1,n,-1),r,q),dim=-1)
        mu, raw = self.forecast_head(feature).chunk(2,dim=-1)
        return mu, F.softplus(raw) + 1e-4
