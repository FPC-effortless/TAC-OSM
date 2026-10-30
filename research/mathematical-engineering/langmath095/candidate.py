"""LANGMATH095 opt-in, no-fit policies over a copied CONTEXT026 descendant.

Historical models and their source remain unchanged. Default policies delegate
to the copied model's original forward. This is a candidate, not a trained or
mathematically certified improvement; full PNDS/source qualification is OPEN.

The source write_mask admits associative-bank updates only. It does not exclude
tokens from other features, waves, concept seeds, or upper recurrent states.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

_CONTEXT = str(Path(__file__).resolve().parents[1] / "context026")
if _CONTEXT not in sys.path:
    sys.path.insert(0, _CONTEXT)
from context_transition import affine_scan, last_write_memory, ordinal_code
from context_transition import phi_position, scalar_affine_scan


@dataclass(frozen=True)
class CandidatePolicies095:
    aligned_write: str = "existing"
    positions: str = "existing"
    concept_graph: str = "existing"
    upper_retention: str = "existing"

    def __post_init__(self):
        choices = {
            "aligned_write": ("existing", "bounded"),
            "positions": ("existing", "bounded"),
            "concept_graph": ("existing", "disabled"),
            "upper_retention": ("existing", "multiscale"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")


def bounded_position_features095(indices, dtype):
    """Replace only log(1+t) by log(1+t)/(1+log(1+t)) in [0,1)."""
    features = phi_position(indices, dtype)
    logarithm = features[..., :1]
    return torch.cat((logarithm / (1 + logarithm), features[..., 1:]), -1)


def bounded_aligned_write095(alpha, raw_write):
    """w=(1-alpha)*sigmoid(g); P=alpha*(I-e*k*k^T).

    For ||k||<=1, 0<=e<=1 and ||v||<=V, ||P||_2<=alpha and
    ||Q||_F<=w*V. Thus ||M||_F<=V is invariant from a zero initial
    state, including changing keys. This is stronger than a constant-key
    equilibrium bound. Low precision and actual-source parity remain separate
    obligations; this policy changes predictions and requires retraining.
    """
    return (1 - alpha) * raw_write.sigmoid()


class TACMathCandidate095(nn.Module):
    """Own a deep copy; state_dict keys acquire the explicit 'model.' prefix.

    Use make_multiscale_candidate095 for the deliberately reinitialized arm.
    Other policy changes are weight-preserving, opt-in checkpoint lesions.
    """

    def __init__(self, baseline, policies=None):
        super().__init__()
        self.policies = policies or CandidatePolicies095()
        if not isinstance(self.policies, CandidatePolicies095):
            raise TypeError("policies must be CandidatePolicies095")
        if self.policies.upper_retention == "multiscale":
            raise ValueError("use make_multiscale_candidate095 for reinitialization")
        self.model = deepcopy(baseline)
        self.retention_initialization = None

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_record(self):
        count = self.parameter_count()
        inactive = 0
        if self.policies.concept_graph == "disabled":
            inactive = sum(p.numel() for part in (self.model.concept_q,
                           self.model.concept_k) for p in part.parameters()
                           if p.requires_grad)
        return {
            "variant": "LANGMATH095", "policies": asdict(self.policies),
            "trainable_parameters": count,
            "deployed_parameters": sum(p.numel() for p in self.parameters()),
            "active_parameters": count - inactive,
            "inactive_graph_parameters": inactive,
            "checkpoint_prefix": "model.",
            "reinitialized": self.retention_initialization is not None,
            "retention_initialization": self.retention_initialization,
            "multiscale_scan_coefficient_precision": "at least FP32",
            "mask_semantics": "associative_banks_only",
            "streaming": False, "trained": False,
            "full_source_pnds_qualification": "OPEN",
        }

    def forward(self, tokens, *, start=0, write_mask=None, return_trace=False):
        if self.policies == CandidatePolicies095():
            return self.model(tokens, start=start, write_mask=write_mask,
                              return_trace=return_trace)
        return self._forward(tokens, start=start, write_mask=write_mask,
                             return_trace=return_trace)

    def _apply_upper(self, module, value):
        if self.policies.upper_retention == "existing":
            return module(value)
        if isinstance(module, nn.Sequential):
            for child in module:
                value = self._apply_upper(child, value)
            return value
        if module.__class__.__name__ == "RecurrenceOnlyUpperBlock":
            u = module.input_proj(module.input_norm(value)).tanh()
            forget = module.forget(u)
            # FP16/BF16 can round exp(-1/4096) to one. Promote before sigmoid;
            # the projection itself still follows autocast, as in the source.
            if forget.dtype in (torch.float16, torch.bfloat16):
                forget = forget.float()
            alpha = forget.sigmoid()  # no historical .995 ceiling
            offer = module.offer(u).tanh()
            state = scalar_affine_scan(alpha, (1 - alpha) * offer)
            prior = F.pad(state[:, :-1], (0, 0, 1, 0))
            gate = module.read_gate(torch.cat((u, prior), -1)).sigmoid()
            return value + module.residual_gain.sigmoid() * module.output(
                (u + gate * prior).tanh())
        if module.__class__.__name__ == "RecurrentFFNBlock":
            z = self._apply_upper(module.recurrence, value)
            u = module.norm(z)
            return z + module.ffn_gain.sigmoid() * module.down(
                module.up(u) * F.silu(module.gate(u)))
        return module(value)

    def _forward(self, tokens, *, start=0, write_mask=None, return_trace=False):
        """Source-order CONTEXT026 forward with only declared policy edits."""
        m = self.model
        if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.size(1) == 0:
            raise ValueError("token tensor must be nonempty [B,T] of int64")
        if type(start) is not int or start != 0:
            raise ValueError("LANGMATH095 requires reset source windows")
        b, t = tokens.shape
        if tokens.min() < 0 or tokens.max() >= m.vocab:
            raise ValueError("token out of range")
        if write_mask is None:
            write_mask = torch.ones(b, t, dtype=torch.bool, device=tokens.device)
        if write_mask.shape != (b, t) or write_mask.dtype != torch.bool:
            raise ValueError("invalid bank admission mask")
        tt = torch.arange(t, device=tokens.device).view(1, t).expand(b, t)
        position = (bounded_position_features095(tt, m.embedding.weight.dtype)
                    if self.policies.positions == "bounded" else
                    phi_position(tt, m.embedding.weight.dtype))
        e = torch.tanh(m.embedding(tokens) + m.pos(position))
        basis = torch.tanh(m.basis(e))
        prev_basis = F.pad(basis.cumsum(1)[:, :-1], (0, 0, 1, 0))
        ctx = prev_basis / tt.clamp_min(1).unsqueeze(-1)
        lexical = torch.tanh(m.embedding(tokens))
        local = lexical
        for lag in range(1, 4):
            past = (F.pad(lexical[:, :-lag], (0, 0, lag, 0))[:, :t]
                    if lag < t else torch.zeros_like(lexical))
            local = local + torch.tanh(m.context_lag_weights[lag - 1]) * past
        key = F.normalize(m.native_key(torch.tanh(local)), dim=-1, eps=1e-6)
        write_key = F.pad(key[:, :-1], (0, 0, 1, 0))
        value = torch.tanh(m.native_value(e))
        phi = ordinal_code(tt, e.dtype)
        aligned_key = F.normalize(torch.einsum("bti,hij->bthj", phi,
                                  m.rotation), dim=-1, eps=1e-6)
        alphabeta = m.native_gate(e).sigmoid()
        aligned_raw = m.aligned_gate(e).view(b, t, 2, 3)
        aligned_gates = aligned_raw.sigmoid()
        allkey = torch.cat((write_key.unsqueeze(2), aligned_key), dim=2)
        alpha = torch.cat((.995 * alphabeta[..., 0:1],
                           aligned_gates[..., 0]), dim=-1)
        erase = torch.cat((alphabeta[..., 1:2], aligned_gates[..., 1]), dim=-1)
        aligned_write = aligned_gates[..., 2]
        if self.policies.aligned_write == "bounded":
            aligned_write = bounded_aligned_write095(aligned_gates[..., 0],
                                                     aligned_raw[..., 2])
        write = torch.cat((alphabeta[..., 1:2], aligned_write), dim=-1)
        allvalue = torch.cat((F.pad(value, (0, 8)).unsqueeze(2),
                    torch.tanh(m.aligned_value(e)).unsqueeze(2).expand(-1, -1, 2, -1)), dim=2)
        eye = torch.eye(8, dtype=e.dtype, device=e.device)
        P = alpha[..., None, None] * (eye - erase[..., None, None] *
                    allkey.unsqueeze(-1) * allkey.unsqueeze(-2))
        Q = write[..., None, None] * allvalue.unsqueeze(-1) * allkey.unsqueeze(-2)
        bank_mask = write_mask[:, :, None].expand(-1, -1, 3).clone()
        bank_mask[:, 0, 0] = False
        P = torch.where(bank_mask[..., None, None], P, eye)
        Q = torch.where(bank_mask[..., None, None], Q, torch.zeros_like(Q))
        mem = affine_scan(P.permute(0, 2, 1, 3, 4).reshape(b * 3, t, 8, 8),
                          Q.permute(0, 2, 1, 3, 4).reshape(b * 3, t, 16, 8))
        mem = mem.reshape(b, 3, t, 16, 8).permute(0, 2, 1, 3, 4)
        mem = last_write_memory(mem.permute(0, 2, 1, 3, 4).reshape(b * 3, t, 16, 8),
                    bank_mask.permute(0, 2, 1).reshape(b * 3, t)).reshape(
                    b, 3, t, 16, 8).permute(0, 2, 1, 3, 4)
        old = F.pad(mem[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))
        native_read = torch.einsum("btij,btj->bti", mem[:, :, 0, :8], key)
        previous = ordinal_code((tt - 1).clamp_min(0), e.dtype)
        previous_key = F.normalize(torch.einsum("bti,hij->bthj", previous,
                                   m.rotation), dim=-1, eps=1e-6)
        aligned_read = torch.einsum("bthvk,bthk->bthv", old[:, :, 1:],
                                    previous_key).mean(2)
        aligned_read = torch.where((tt > 0).unsqueeze(-1), aligned_read,
                                   torch.zeros_like(aligned_read))
        angle = m.wave_angle.to(dtype=e.dtype)
        c = angle.cos().view(1, 1, 8)
        s = angle.sin().view(1, 1, 8)
        rho = .97 + .029 * m.wave_decay(e).sigmoid()
        R = torch.stack((torch.stack((c.expand(b, t, -1), -s.expand(b, t, -1)), -1),
                         torch.stack((s.expand(b, t, -1), c.expand(b, t, -1)), -1)), -2)
        p_wave = rho[..., None, None] * R
        q_wave = .025 * torch.tanh(m.wave_input(e)).view(b, t, 8, 1, 2)
        wave = affine_scan(p_wave.permute(0, 2, 1, 3, 4).reshape(b * 8, t, 2, 2),
                           q_wave.permute(0, 2, 1, 3, 4).reshape(b * 8, t, 1, 2))
        wave = wave.view(b, 8, t, 2).permute(0, 2, 1, 3)
        oldwave = F.pad(wave[:, :-1], (0, 0, 0, 0, 1, 0)).flatten(-2)
        cg = .05 * m.concept_gate(e).sigmoid()
        cp = scalar_affine_scan(1 - cg, cg * torch.tanh(m.concept_value(e))).view(b, t, 6, 16)
        oldcp = F.pad(cp[:, :-1], (0, 0, 0, 0, 1, 0))
        f = torch.tanh(m.fast(torch.cat((e, ctx, oldwave, native_read), -1)))
        slow = torch.tanh(m.slow(torch.cat((f, ctx, native_read), -1)))
        G = torch.tanh(m.evidence(torch.cat((e, f, slow, aligned_read,
                                           oldwave), -1))).view(b, t, 6, 16)
        if self.policies.concept_graph == "disabled":
            L = torch.zeros(b, t, 6, 6, dtype=G.dtype, device=G.device)
        else:
            q, k = m.concept_q(G), m.concept_k(G)
            sm = torch.softmax(q @ k.transpose(-1, -2) / 4, -1)
            A = .5 * (sm + sm.transpose(-1, -2))
            A = A * (1 - torch.eye(6, dtype=A.dtype, device=A.device))
            L = torch.diag_embed(A.sum(-1)) - A
        eye6 = torch.eye(6, dtype=e.dtype, device=e.device)
        H = 1.25 * eye6 + .25 * L
        C = torch.linalg.solve(H, G + .25 * oldcp)
        pooling = torch.softmax(G.mean(-1), dim=-1)
        pooled = (C * pooling.unsqueeze(-1)).sum(dim=-2)
        z = self._apply_upper(m.layernorm, torch.cat((f, slow, native_read,
                              aligned_read, oldwave, pooled, e), -1))
        logits = m.ffn_out(m.ffn_up(z) * F.silu(m.ffn_gate(z))) + m.concept_out(pooled)
        if not return_trace:
            return logits
        return logits, dict(native_prewrite=old[:, :, 0, :8],
            native_memory=mem[:, :, 0, :8], native_read=native_read,
            native_query_key=key, native_write_key=write_key, native_value=value,
            native_alpha=alpha[..., 0], native_beta=write[..., 0],
            native_admission=bank_mask[..., 0], aligned_prewrite=old[:, :, 1:],
            wave_prewrite=oldwave, concept_prewrite=oldcp, concept_posterior=C,
            concept_laplacian=L, gate_beta=write, basis=ctx,
            aligned_alpha=alpha[..., 1:], aligned_write=aligned_write,
            position_features=position, concept_evidence=G, concept_pooling=pooling)


def make_multiscale_candidate095(baseline, policies=None, *, min_horizon=2,
                                 max_horizon=4096):
    """New-initialization arm: alpha_i=exp(-1/tau_i), log-spaced tau.

    Remove the .995 ceiling only in this candidate's upper dispatch. Zero the
    copied forget weights and initialize bias=logit(alpha). All parameters
    remain trainable. This is explicitly not a trained-checkpoint lesion.
    """
    if not (math.isfinite(min_horizon) and math.isfinite(max_horizon)
            and 0 < min_horizon <= max_horizon):
        raise ValueError("finite positive ordered horizons required")
    policies = policies or CandidatePolicies095()
    candidate = TACMathCandidate095(baseline,
                    replace(policies, upper_retention="existing"))
    blocks = [m for m in candidate.model.modules()
              if m.__class__.__name__ == "RecurrenceOnlyUpperBlock"]
    if not blocks:
        raise ValueError("baseline has no supported upper recurrent blocks")
    with torch.no_grad():
        for block in blocks:
            tau = torch.logspace(math.log10(min_horizon), math.log10(max_horizon),
                block.state_width, dtype=block.forget.bias.dtype,
                device=block.forget.bias.device)
            alpha = torch.exp(-1 / tau)
            if not bool(((alpha > 0) & (alpha < 1)).all()):
                raise ValueError("horizons are not representable in parameter dtype")
            block.forget.weight.zero_()
            block.forget.bias.copy_(torch.logit(alpha))
    candidate.policies = replace(policies, upper_retention="multiscale")
    candidate.retention_initialization = {
        "minimum_e_fold_horizon": min_horizon,
        "maximum_e_fold_horizon": max_horizon,
        "upper_blocks": len(blocks), "forget_weights_zeroed": True,
        "historical_ceiling_removed": True,
    }
    return candidate
