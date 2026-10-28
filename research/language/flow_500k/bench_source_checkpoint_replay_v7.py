"""GRAPH011: source-verified, sparse recurrent-state checkpoints + on-demand replay.

Inspired by the *time-space tradeoff principle* discussed in R. Ryan
Williams (STOC 2025), NOT an implementation of his universal simulation
or the Cook-Mertz catalytic tree algorithm.

Store ORIGINAL observed bytes plus every K-th complete frozen model state.
For a query after token t, reload the greatest checkpoint strictly before t
and replay at most K source tokens. No future labels or model-generated
events are invented. In-memory source bytes, checkpoint bytes, model weights,
recomputation tokens, and elapsed time are reported SEPARATELY.

Correctness requires a deterministic frozen model, verified source bytes,
replayable streaming state, identical execution settings and an authorized
owner; model/CPU floating roundoff may prevent bit-identical logits.
"""
from __future__ import annotations

import copy
import hashlib
import io
import time
from dataclasses import dataclass
from typing import Any

import torch


def _snapshot(state: Any) -> Any:
    if isinstance(state, torch.Tensor):
        return state.detach().clone()
    if isinstance(state, dict):
        return {k: _snapshot(v) for k, v in state.items()}
    if isinstance(state, tuple):
        return tuple(_snapshot(x) for x in state)
    if isinstance(state, list):
        return [_snapshot(x) for x in state]
    return copy.deepcopy(state)


def _serialized_bytes(item: Any) -> int:
    buffer = io.BytesIO()
    torch.save(item, buffer)
    return buffer.tell()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _model_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, param in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(param.shape)).encode("ascii"))
        digest.update(str(param.dtype).encode("ascii"))
        raw = io.BytesIO()\n        torch.save(param.detach().cpu().contiguous(), raw)\n        digest.update(raw.getvalue())
    return digest.hexdigest()


@dataclass(frozen=True)
class ReplayReceipt:
    owner: str
    source_sha256: str
    model_sha256: str
    checkpoint_ordinal: int
    target_ordinal: int
    replayed_tokens: int
    checkpoint_bytes: int
    source_bytes: int
    model_weight_bytes: int
    seconds: float


class VerifiedSparseReplay:
    def __init__(self, model, observed_source: bytes, *, owner: str,
                 block: int, source_sha256: str):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("frozen streaming torch module required")
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("use eval and freeze parameters before replay")
        if type(owner) is not str or not owner or type(block) is not int or block < 1:
            raise ValueError("nonempty owner and positive integer block required")
        if type(observed_source) is not bytes or not observed_source:
            raise ValueError("original observed source bytes required")
        if any(byte >= 128 for byte in observed_source):
            raise ValueError("source must use exact 128-byte model vocabulary; map high bytes transparently before use")
        if type(source_sha256) is not str or len(source_sha256) != 64 or (
                _sha(observed_source) != source_sha256):
            raise ValueError("source byte hash does not match pinned manifest")
        self.model = model
        self.source = observed_source
        self.owner = owner
        self.block = block
        self.source_sha256 = source_sha256
        self.weights_sha256 = _model_digest(model)
        self.model_weight_bytes = sum(p.numel()*p.element_size()
                                      for p in model.parameters())
        self.checkpoints: dict[int, Any] = {0: None}
        self.prepared = False

    def _source_tensor(self, start: int, stop: int) -> torch.Tensor:
        data = list(self.source[start:stop])
        if not data:
            raise ValueError("cannot pass empty stream to source model")
        return torch.tensor(data, dtype=torch.long).view(1, -1)

    def _step(self, start: int, stop: int, state: Any):
        with torch.inference_mode():
            prob, newstate = self.model(self._source_tensor(start, stop),
                start=start, source_ids=(self.owner,),
                initial_state=_snapshot(state), return_state=True)
        if prob.shape != (1, stop-start, 128) or not bool(torch.isfinite(prob).all()):
            raise ValueError("source model returned invalid categorical result")
        if not bool(torch.allclose(prob.sum(-1),
                torch.ones_like(prob[..., 0]), atol=3e-5)):
            raise ValueError("source model failed categorical probability sum")
        if newstate.get("count") != stop or newstate.get("source_ids") != (self.owner,):
            raise ValueError("source model advanced incorrect ordinal/owner")
        return prob, newstate

    def prepare(self):
        if self.prepared:
            raise ValueError("checkpoints already prepared")
        if _model_digest(self.model) != self.weights_sha256:
            raise ValueError("frozen checkpoint parameters changed")
        n = len(self.source)
        state = None
        # Build once, using only observed source. Retain one snapshot per
        # block instead of one snapshot per event.
        for start in range(0, n, self.block):
            stop = min(start+self.block, n)
            _, state = self._step(start, stop, state)
            if stop < n and stop % self.block == 0:
                self.checkpoints[stop] = _snapshot(state)
        self.prepared = True
        return self

    def resident_checkpoint_bytes(self) -> int:
        if not self.prepared:
            raise ValueError("prepare before measuring")
        # Actual torch serialization bytes, including headers & Python
        # queue data. This is storage, not peak device activation RAM.
        return sum(_serialized_bytes(st) for st in self.checkpoints.values())

    def predict_at(self, target: int, *, owner: str, expected_sha256: str):
        """Predict the next symbol after observed source byte at t-1.

        target is count of observed bytes, not an unobserved label.
        The state checkpoint at t itself is excluded, so at least one
        real-source byte passes through the model and produces logits.
        """
        if not self.prepared:
            raise ValueError("unprepared index")
        if owner != self.owner:
            raise PermissionError("cross-owner replay denied")
        if expected_sha256 != self.source_sha256 or _sha(self.source) != self.source_sha256:
            raise ValueError("source identity/hash mismatch")
        if _model_digest(self.model) != self.weights_sha256:
            raise ValueError("model weight digest mismatch; source replay invalid")
        if type(target) is not int or not (1 <= target <= len(self.source)):
            raise ValueError("target must be observed positive prefix length")
        cp = ((target - 1)//self.block)*self.block
        if cp not in self.checkpoints:
            raise ValueError("missing nearest source checkpoint")
        begin = time.perf_counter()
        logits, state = self._step(cp, target, self.checkpoints[cp])
        elapsed = time.perf_counter()-begin
        r = ReplayReceipt(
            owner=owner, source_sha256=self.source_sha256,
            model_sha256=self.weights_sha256,
            checkpoint_ordinal=cp, target_ordinal=target,
            replayed_tokens=target-cp,
            checkpoint_bytes=self.resident_checkpoint_bytes(),
            source_bytes=len(self.source),
            model_weight_bytes=self.model_weight_bytes,
            seconds=elapsed)
        if r.replayed_tokens > self.block:
            raise AssertionError("replay exceeded one block")
        return logits[:, -1, :], _snapshot(state), r
