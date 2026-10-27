"""TAC-LANG-REPAIR-016: parameter-free candidate numerical repair operators.

Research-only. The *frozen* 100,680-parameter language graph and all original
checkpoints are UNTOUCHED. These functions can be injected into a new,
separately trained arm after preregistration, matched controls and lawful
real-text dataset authorization. No inference substitution is promoted.

Problems:
  - hard F.normalize(eps=1e-6) near zero has slope 1e6 and kinks at norm=eps;
  - the original 7,13,29,59 phase tuple repeats every 155701 ordinals;
  - arbitrary learned 8x8 rotation may collapse the source phase vector.

Candidate:
  - sqrt(||x||²+tau²) smooth key with Jacobian operator norm <= 1/tau;
  - retain source exact phase within era 0 and tag later eras;
  - bounded rotation near a fixed orthogonal base with a provable
    sigma_min >= 1 - 8*delta, for 8x8 matrices and delta<1/8.
"""
from __future__ import annotations
import math
import torch

PERIODS=(7,13,29,59)
POSITION_CYCLE=7*13*29*59

def smooth_key(x: torch.Tensor,tau: float=0.01) -> torch.Tensor:
    if tau<=0 or not math.isfinite(tau):
        raise ValueError("positive finite smoothing radius required")
    if x.shape[-1]!=8 or not torch.is_floating_point(x):
        raise ValueError("expected floating 8-d key")
    if not torch.isfinite(x).all():
        raise ValueError("nonfinite key")
    return x / torch.sqrt(x.square().sum(-1,keepdim=True)+tau*tau)

def phase_code(ordinal:int,dtype:torch.dtype=torch.float64,
               device:torch.device|str="cpu") -> torch.Tensor:
    if not isinstance(ordinal,int) or ordinal<0 or ordinal>2**63-1:
        raise ValueError("nonnegative signed-int64 ordinal required")
    # Integer remainders BEFORE float conversion to avoid lost low bits.
    parts=[]
    for p in PERIODS:
        a=torch.tensor((ordinal%p)*(2*math.pi/p),dtype=dtype,device=device)
        parts.extend((torch.cos(a),torch.sin(a)))
    return torch.stack(parts,-1)

def epoch_code(ordinal:int,dtype:torch.dtype=torch.float64,
               device:torch.device|str="cpu",era_scale:float=0.25) -> torch.Tensor:
    if not (0<era_scale<=0.5):
        raise ValueError("era scale required in (0,.5]")
    base=phase_code(ordinal,dtype,device)
    era=ordinal//POSITION_CYCLE
    # Era=0 EXACTLY reproduces original phase code. Later era exact-real
    # differences break forced same-residue aliases (not arbitrary collisions).
    # Finite floats can still round nearby large eras together.
    base[0]=base[0]+era_scale*(era/(era+1))
    return base

def conditioned_rotation(raw:torch.Tensor,delta:float=0.05)->torch.Tensor:
    if not (0<delta<1/8):
        raise ValueError("delta must be positive and below 1/8")
    if raw.shape!= (2,8,8) or not torch.is_floating_point(raw):
        raise ValueError("requires two learned 8x8 rotation matrices")
    if not torch.isfinite(raw).all():
        raise ValueError("nonfinite rotation")
    eye=torch.eye(8,device=raw.device,dtype=raw.dtype)
    # Base exactly matches original source initialization.
    base=torch.stack((eye,torch.roll(eye,1,1)))
    # Every learned coordinate contributes continuously, but all singular
    # values are bounded below by 1-8*delta from Frobenius <= 8*delta.
    return base+delta*torch.tanh(raw-base)

def repaired_source_key(raw:torch.Tensor,ordinal:int,tau:float=0.01,
                        era_scale:float=0.25,delta:float=0.05)->torch.Tensor:
    phi=epoch_code(ordinal,raw.dtype,raw.device,era_scale)
    r=conditioned_rotation(raw,delta)
    return smooth_key(torch.einsum("i,hij->hj",phi,r),tau)
