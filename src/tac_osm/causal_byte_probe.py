"""SENSE-TOK-002 source-only matched-parameter byte predictor preflight.

No optimizer/training is provided or permitted by this module. Every next-byte
input must be built strictly before observing its byte target. Parameter
counts are from live PyTorch model objects, not design estimates.
"""
from __future__ import annotations

from .causal_patch_stream import CausalPatchStream, PrefixView, StreamVariant
from .sensory_tokenizers import _classes

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:
    raise ImportError("causal_byte_probe requires torch on an authorized compute runner") from exc

BOS_ID = 256
FEATURE_DIM = 16


def features_from_view(view: PrefixView) -> tuple[float, ...]:
    """Canonical transparent representation of already observed bytes."""
    view.validate()
    bucket = min(len(view.pending), 7)
    lengths = [1.0 if bucket == k else 0.0 for k in range(8)]
    last_byte = view.complete_byte_payload[-1] if view.observed else None
    classes = ([1.0 if _classes(last_byte) == k else 0.0 for k in range(1, 6)]
               if last_byte is not None else [0.0] * 5)
    last_length = len(view.committed[-1].patch.payload) if view.committed else 0
    return tuple(lengths + classes + [
        last_length / 24.0,
        len(view.committed) / max(1, view.observed),
        float(bool(view.committed and view.committed[-1].available_after == view.observed)),
    ])


def quick_features(stream: CausalPatchStream) -> tuple[float, ...]:
    """O(1) feature path tested for equivalence against the reviewed prefix view."""
    bucket = min(len(stream._pending), 7)
    length_onehot = [1.0 if i == bucket else 0.0 for i in range(8)]
    last = stream._observed[-1] if stream._observed else None
    classes = [float(last is not None and _classes(last) == k) for k in range(1, 6)]
    completed = stream._committed
    last_len = len(completed[-1].patch.payload) if completed else 0
    return tuple(length_onehot + classes + [
        last_len / 24.0,
        len(completed) / max(1, stream.observed_count),
        float(bool(completed and completed[-1].available_after == stream.observed_count)),
    ])


class CausalByteGRU(nn.Module):
    """A shared 100,400-parameter unfitted backbone for T0/T1/T2/T2b."""

    def __init__(self) -> None:
        super().__init__()
        self.byte_emb = nn.Embedding(257, 32)
        self.gru = nn.GRU(input_size=48, hidden_size=120, batch_first=True)
        self.byte_head = nn.Linear(120, 256)

    def forward(self, prev_ids: Tensor, patch_features: Tensor) -> Tensor:
        if prev_ids.ndim != 2 or patch_features.ndim != 3:
            raise ValueError("expected [batch,time] IDs and [batch,time,16] features")
        if patch_features.shape != (*prev_ids.shape, FEATURE_DIM):
            raise ValueError("different time axis or feature dimension")
        if patch_features.dtype != torch.float32:
            raise TypeError("features must be float32")
        hidden, _ = self.gru(torch.cat([self.byte_emb(prev_ids), patch_features], dim=-1))
        return self.byte_head(hidden)


def predecision_examples(raw: bytes, variant: StreamVariant) -> tuple[Tensor, Tensor, Tensor]:
    """Encode x[:t] as inputs; only after that reveal target byte x[t]."""
    if not isinstance(raw, bytes):
        raise TypeError("raw must be bytes")
    stream = CausalPatchStream(variant=variant)
    ids: list[int] = []
    features: list[tuple[float, ...]] = []
    for i, byte in enumerate(raw):
        ids.append(BOS_ID if i == 0 else raw[i - 1])
        features.append(quick_features(stream))
        stream.observe(bytes([byte]))
    if not raw:
        return (torch.empty(0, dtype=torch.long),
                torch.empty((0, FEATURE_DIM), dtype=torch.float32),
                torch.empty(0, dtype=torch.long))
    return (torch.tensor(ids, dtype=torch.long),
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(list(raw), dtype=torch.long))


def parameter_receipt(model: nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    deployed = sum(p.numel() for p in model.parameters())
    return {
        "trainable_parameters": trainable,
        "deployed_learned_parameters": deployed,
        "absolute_100k_band": 95_000 <= trainable <= 105_000 and 95_000 <= deployed <= 105_000,
        "parameterized_components": {name: p.numel() for name, p in model.named_parameters()},
        "training_authorized": False,
    }


class LearnedSensoryGateGRU(nn.Module):
    """Untrained soft sensory gate, 100,513 parameters; NOT discrete tokenization.

    g_t uses only previous byte embedding and previous recurrent state.
    m_t = (1-g_t)*m_(t-1) + g_t*e_(t-1).
    Every byte executes the gate and GRUCell. No compute savings established.
    """

    def __init__(self) -> None:
        super().__init__()
        self.byte_emb = nn.Embedding(257, 32)
        self.boundary_gate = nn.Linear(32 + 115, 1)
        self.gru_cell = nn.GRUCell(input_size=64, hidden_size=115)
        self.byte_head = nn.Linear(115, 256)

    def forward(self, prev_ids: Tensor, *, return_gates: bool = False):
        if prev_ids.ndim != 2:
            raise ValueError("expected [batch,time] previous byte IDs")
        b, t = prev_ids.shape
        emb = self.byte_emb(prev_ids)
        h = emb.new_zeros((b, 115))
        local = emb.new_zeros((b, 32))
        logits: list[Tensor] = []
        gates: list[Tensor] = []
        for k in range(t):
            g = torch.sigmoid(self.boundary_gate(torch.cat([emb[:, k], h], dim=-1)))
            local = (1 - g) * local + g * emb[:, k]
            h = self.gru_cell(torch.cat([emb[:, k], local], dim=-1), h)
            logits.append(self.byte_head(h))
            gates.append(g)
        out = torch.stack(logits, dim=1) if logits else emb.new_empty((b, 0, 256))
        if return_gates:
            return out, torch.stack(gates, dim=1) if gates else emb.new_empty((b, 0, 1))
        return out
