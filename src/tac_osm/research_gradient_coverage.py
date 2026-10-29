"""Live-model coverage for research backward parity; not an AD correctness proof.

Capture before executing either arm. Equality of two partial gradient dictionaries
does not establish coverage. A zero tensor is present; a missing tensor is not.
Shared parameter objects count once, as in PyTorch's named_parameters().
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import torch


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    header = json.dumps([list(value.shape), str(value.dtype)], separators=(",", ":"))
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header.encode() + b"\0" + raw).hexdigest()


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    identity: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    trainable: bool
    elements: int


@dataclass(frozen=True)
class GradientContract:
    model_identity: int
    parameters: tuple[ParameterSpec, ...]
    state_digests: tuple[tuple[str, str], ...]

    @property
    def trainable(self) -> tuple[ParameterSpec, ...]:
        return tuple(p for p in self.parameters if p.trainable)

    @property
    def elements(self) -> int:
        return sum(p.elements for p in self.trainable)

    @classmethod
    def capture(cls, model: torch.nn.Module) -> "GradientContract":
        specs = tuple(ParameterSpec(n, id(p), tuple(p.shape), p.dtype, p.device,
                                    p.requires_grad, p.numel())
                      for n, p in model.named_parameters())
        if not any(p.trainable for p in specs):
            raise ValueError("no declared trainable parameter coverage")
        return cls(id(model), specs, tuple((n, tensor_sha256(t))
                                          for n, t in model.state_dict().items()))

    def verify_model(self, model: torch.nn.Module) -> None:
        current = GradientContract.capture(model)
        if self.model_identity != current.model_identity or self.parameters != current.parameters:
            raise ValueError("live parameter identity/schema/trainability changed")
        if self.state_digests != current.state_digests:
            raise ValueError("model weights or buffers changed during no-fit audit")

    def validate(self, model: torch.nn.Module, result: dict,
                 *, logits_shape: tuple[int, ...]) -> dict:
        """Fail closed independently of the other comparison arm's contents."""
        self.verify_model(model)
        if result.get("missing") != []:
            raise ValueError("missing gradient declarations are not complete coverage")
        grads = result.get("grads")
        expected = {p.name: p for p in self.trainable}
        if not isinstance(grads, dict) or set(grads) != set(expected):
            raise ValueError("gradient coverage differs from live trainable parameter schema")
        for name, shape in (("logits", logits_shape), ("loss", ())):
            value = result.get(name)
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError("invalid finite exact-shape " + name)
        rows = []
        for name, spec in expected.items():
            grad = grads[name]
            if (not isinstance(grad, torch.Tensor) or grad.layout != torch.strided
                    or tuple(grad.shape) != spec.shape or grad.dtype != spec.dtype
                    or grad.device != spec.device or not torch.isfinite(grad).all()):
                raise ValueError("invalid gradient shape/dtype/device/finiteness: " + name)
            rows.append({"name": name, "shape": list(spec.shape),
                         "dtype": str(spec.dtype), "elements": grad.numel(),
                         "nonzero_elements": int(torch.count_nonzero(grad)),
                         "sha256": tensor_sha256(grad)})
        return {"status": "PASS_COMPLETE_DECLARED_TRAINABLE_GRADIENTS",
                "tensor_count": len(rows), "element_count": sum(r["elements"] for r in rows),
                "parameters": rows,
                "limits": "Presence/finite numerical parity is not proof of correct AD, useful learning, no dummy weights, reward integrity or independent qualification"}
