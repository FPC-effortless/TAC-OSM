"""Fail-closed session contracts for frozen TAC-LANG-GATE-003 source.

Source language model is unchanged. This adapter enforces known ordinal,
tokenizer, owner, finite-state, and trained-geometry preconditions only.
Software checks are not whole-program IEEE or Lean proofs.
"""
from __future__ import annotations
import math
import torch

VOCAB_SIZE = 128
PERIODS = (7, 13, 29, 59)
FORCED_KEY_PERIOD = math.lcm(*PERIODS)
EXPECTED_ACTIVE_PARAMETERS = 100_680
EXPECTED_FLOAT_STATE = 88 + 8*8 + 2*16*8

# Complete source-to-mathematics live tensor-shape manifest (36 tensor groups).
# Flattened parameter totals alone cannot detect swapped or inactive layers.
EXPECTED_PARAMETER_SHAPES = {
    "rotation": (2,8,8), "res_logit": (),
    "embedding.weight": (128,48), "time_proj.weight": (48,7),
    "key_native.weight": (8,48), "key_native.bias": (8,),
    "value_native.weight": (8,48), "value_native.bias": (8,),
    "fast_map.weight": (176,144), "fast_map.bias": (176,),
    "slow_map.weight": (176,184), "slow_map.bias": (176,),
    "native_gates.weight": (2,224), "native_gates.bias": (2,),
    "source_content.weight": (16,48), "source_content.bias": (16,),
    "source_router.weight": (2,48), "source_router.bias": (2,),
    "source_retention.weight": (2,48), "source_retention.bias": (2,),
    "source_strength.weight": (2,48), "source_strength.bias": (2,),
    "source_silu_mod.weight": (2,48),
    "hist_project.weight": (48,16),
    "read_router.weight": (2,8), "read_router.bias": (2,),
    "norm.weight": (248,), "norm.bias": (248,),
    "read_in.weight": (75,248), "read_in.bias": (75,),
    "read_out.weight": (128,75), "read_out.bias": (128,),
    "res_in.weight": (10,248), "res_in.bias": (10,),
    "res_out.weight": (128,10), "res_out.bias": (128,),
}

def require_ordinal_range(first: int, count: int) -> None:
    """No forced modulo cycle wrap; NOT a proof of unique learned keys."""
    if not isinstance(first,int) or isinstance(first,bool) or first < 0:
        raise ValueError("ordinal must be a nonnegative Python integer")
    if not isinstance(count,int) or isinstance(count,bool) or count < 1:
        raise ValueError("count must be a positive Python integer")
    if first + count > FORCED_KEY_PERIOD:
        raise ValueError("source ordinal Fourier address is outside its first period")

def validate_token(token: torch.Tensor, batch: int | None = None) -> None:
    if not isinstance(token,torch.Tensor) or token.dtype != torch.int64:
        raise ValueError("token must be torch.int64")
    if token.ndim not in (1,2) or (batch is not None and token.shape[0] != batch):
        raise ValueError("invalid token shape or batch ownership")
    if not (torch.all(token >= 0) and torch.all(token < VOCAB_SIZE)):
        raise ValueError("token outside explicit 128-character vocabulary")

def validate_model(model) -> None:
    if getattr(model,"decoder",None) != 75 or getattr(model,"post",None) != 10:
        raise ValueError("frozen experiment needs decoder=75, post=10")
    # Source train_gate.py: make(mode) explicitly uses multiscale, NOT the
    # class default multiscale_relative. Both have identical parameter counts.
    if getattr(model,"position_mode",None) != "multiscale":
        raise ValueError("frozen checkpoint used multiscale (one-lag) positional mode")
    if getattr(model,"optimized_scan",None) is not True:
        raise ValueError("frozen checkpoint used the shared rectangular scan")
    if getattr(model,"gate_mode",None) not in ("tanh","silu"):
        raise ValueError("unrecognized historical training arm")
    observed={name:tuple(p.shape) for name,p in model.named_parameters()}
    if observed != EXPECTED_PARAMETER_SHAPES:
        raise ValueError("language trainable tensor shapes/names differ from typed PNDS source graph")
    n=sum(p.numel() for p in model.parameters())
    if n != EXPECTED_ACTIVE_PARAMETERS:
        raise ValueError(f"wrong graph parameter count: {n}")
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("nonfinite model parameters")

def validate_state(state, batch: int, owner: torch.Tensor) -> None:
    if not isinstance(state,dict) or not all(k in state for k in
        ("sum","native","aligned","count","owner")):
        raise ValueError("missing language state field")
    if state["sum"].shape != (batch,88) or state["native"].shape != (batch,8,8) or state["aligned"].shape != (batch,2,16,8):
        raise ValueError("state shape mismatch")
    if not isinstance(state["count"],int) or state["count"] < 0:
        raise ValueError("invalid counter")
    if owner.dtype != torch.int64 or owner.shape != (batch,) or not torch.equal(state["owner"],owner):
        raise ValueError("cross-owner state transfer refused")
    if not all(torch.isfinite(state[k]).all() for k in ("sum","native","aligned")):
        raise ValueError("nonfinite state")

class CheckedLanguageSession:
    """Owner-locked, chronological streaming wrapper, not a new model."""
    def __init__(self,model,owner:torch.Tensor):
        validate_model(model)
        if not isinstance(owner,torch.Tensor) or owner.ndim != 1 or owner.dtype != torch.int64:
            raise ValueError("owner must be int64[B]")
        self.model=model
        self.owner=owner.clone()
        self.state=model.init_stream(owner.numel(),owner=owner)
        validate_state(self.state,owner.numel(),self.owner)

    def step(self,token:torch.Tensor):
        validate_token(token,self.owner.numel())
        if token.ndim != 1:raise ValueError("step expects one character per owner")
        validate_state(self.state,self.owner.numel(),self.owner)
        require_ordinal_range(self.state["count"],1)
        logits,next_state=self.model.stream_step(token,self.state,owner=self.owner)
        validate_state(next_state,self.owner.numel(),self.owner)
        if next_state["count"] != self.state["count"]+1:
            raise ValueError("chronological counter did not advance by one")
        if logits.shape != (self.owner.numel(),VOCAB_SIZE) or not torch.isfinite(logits).all():
            raise ValueError("nonfinite or invalid categorical logits")
        self.state=next_state
        return logits

    def categorical(self,token:torch.Tensor):
        p=torch.softmax(self.step(token),dim=-1)
        if not torch.isfinite(p).all() or (p < 0).any():
            raise ValueError("nonfinite categorical output")
        return p

def running_mean_update(mean:torch.Tensor,basis:torch.Tensor,count:int)->torch.Tensor:
    """Proposed replacement for unbounded source sum, NOT checkpoint-qualified.
    Exact real mean_(n+1)=mean_n+(basis_n-mean_n)/(n+1).
    """
    if not isinstance(count,int) or count<0 or mean.shape != basis.shape:
        raise ValueError("bad running mean inputs")
    return mean+(basis-mean)/(count+1)
