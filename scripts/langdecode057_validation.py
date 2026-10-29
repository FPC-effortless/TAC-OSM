"""Batch finite-state checks around the unchanged LANGSTREAM043 transition.

Inference only. State schema, snapshots, model checks and transition arithmetic
are inherited unchanged. The bounded temporary concatenation is not retained.
"""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/mathematical-engineering/langstream043"))
from incremental import InferenceState, IncrementalTAC043, MAX_CLOCK, SHAPES


class BatchedValidationTAC057(IncrementalTAC043):
    def validate(self, state, owners):
        self._check_model()
        owners = self._owners(owners)
        if type(state) is not InferenceState or state.owners != owners:
            raise ValueError("state owner mismatch")
        if (state.weights_sha256 != self.weights_sha256
                or state.position_mode != self.position_mode
                or state.implementation_sha256 != self.implementation_sha256):
            raise ValueError("state model/mode mismatch")
        if type(state.clock) is not int or not 0 <= state.clock < MAX_CLOCK:
            raise ValueError("invalid state clock")
        if set(state.tensors) != set(SHAPES):
            raise ValueError("state schema mismatch")
        weight = self.model.embedding.weight
        flat = []
        for name, shape in SHAPES.items():
            value = state.tensors[name]
            if (not torch.is_tensor(value)
                    or tuple(value.shape) != (len(owners), *shape)
                    or value.dtype != weight.dtype or value.device != weight.device
                    or value.requires_grad):
                raise ValueError("state tensor invalid: " + name)
            flat.append(value.reshape(-1))
        # All fields have the same checked dtype/device. Finiteness of their
        # concatenation is equivalent to the conjunction of per-field checks.
        # Keep the slow diagnostic path to identify the original bad field.
        if not torch.isfinite(torch.cat(flat)).all():
            for name, value in state.tensors.items():
                if not torch.isfinite(value).all():
                    raise ValueError("state tensor invalid: " + name)
