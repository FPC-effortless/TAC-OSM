#!/usr/bin/env python3
"""Bounded-forward native-state theorem premises checked on original pinned code.

Adversarial mechanism diagnostics ONLY, NOT real-world skill or trained
checkpoint verification. Covers native full fast/slow/associative state under
writable/read-only events, and exact read-only noncontraction in memory
coordinates. Does not prove a neural Lipschitz spectral norm upper bound,
gradient stability through 1000 updates, or useful recall after distractors.

Original source loaded with verified Git blob at
081e534a074001c6a3b2251b3ff31ad20ec4fca8.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F
from verify_original_full_recurrence import native_step
from verify_pinned_source import modules


def assert_bounds(f, s, m, *, dtype, step):
    eps = 1e-6 if dtype == torch.float32 else 1e-10
    if not (torch.isfinite(f).all() and torch.isfinite(s).all()
            and torch.isfinite(m).all()):
        raise RuntimeError(f"nonfinite full native state at {step}")
    if max(f.abs().max().item(), s.abs().max().item()) > 1 + eps:
        raise RuntimeError(f"native fast/slow unit cube violated at {step}")
    norms = torch.linalg.vector_norm(m.reshape(m.shape[0], -1), dim=-1)
    if norms.max().item() > 4 + eps:
        raise RuntimeError(f"associative Frobenius radius violated at {step}")


def verify():
    torch.set_num_threads(1)
    source, _ = modules()
    torch.manual_seed(20261024)
    cases = 0
    peak = 0.0
    for dtype in (torch.float32, torch.float64):
        model = source.AlignedAssociativeTAC().to(dtype=dtype).eval()
        with torch.no_grad():
            # Stress larger learned weights without changing original architecture;
            # generated parameters are NOT trained model/checkpoint evidence.
            for name, parameter in model.named_parameters():
                if "weight" in name and parameter.is_floating_point():
                    parameter.mul_(2.0)
            for length in (1, 32, 512):
                batch = 2
                f = torch.zeros(batch, model.width, dtype=dtype)
                s = f.clone()
                m = torch.zeros(batch, 8, 8, dtype=dtype)
                for step in range(length):
                    kind = torch.tensor([step % 3, (step + 2) % 3])
                    encoded = torch.randn(batch, 48, dtype=dtype).tanh()
                    address = F.normalize(torch.randn(batch, 8, dtype=dtype), dim=-1)
                    content = F.normalize(torch.randn(batch, 8, dtype=dtype), dim=-1)
                    old_m = m.clone()
                    f, s, m, gates = native_step(
                        model, kind, encoded, address, content, f, s, m, source)
                    assert_bounds(f, s, m, dtype=dtype, step=step)
                    if not torch.equal(m[kind == 2], old_m[kind == 2]):
                        raise RuntimeError(f"read-only associative memory altered at {step}")
                    if not ((gates >= 0).all() and (gates <= 1).all()):
                        raise RuntimeError("memory gates outside [0,1]")
                    peak = max(peak, torch.linalg.vector_norm(
                        m.reshape(batch, -1), dim=-1).max().item())
                    cases += 1

        # An exact read-only memory-coordinate perturbation has gain >= 1
        # in the full Euclidean state for ANY learnt fast/slow parameters.
        f = torch.randn(1, 88, dtype=dtype) * .2
        s = torch.randn(1, 88, dtype=dtype) * .2
        m = torch.randn(1, 8, 8, dtype=dtype) * .02
        vector = torch.zeros(240, dtype=dtype)
        vector[176:] = F.normalize(torch.randn(64, dtype=dtype), dim=0)
        state = torch.cat((f.flatten(), s.flatten(), m.flatten()))
        kind = torch.tensor([2])
        event = torch.randn(1, 48, dtype=dtype).tanh()
        addr = F.normalize(torch.randn(1, 8, dtype=dtype), dim=-1)
        val = F.normalize(torch.randn(1, 8, dtype=dtype), dim=-1)

        def step_fn(z):
            ff = z[:88].reshape(1, 88)
            ss = z[88:176].reshape(1, 88)
            mm = z[176:].reshape(1, 8, 8)
            nf, ns, nm, _ = native_step(
                model, kind, event, addr, val, ff, ss, mm, source)
            return torch.cat((nf.flatten(), ns.flatten(), nm.flatten()))

        output, tangent = torch.autograd.functional.jvp(
            step_fn, state, vector, create_graph=False)
        eps = 2e-6 if dtype == torch.float64 else 1e-5
        torch.testing.assert_close(tangent[176:], vector[176:],
                                   rtol=0, atol=eps)
        if torch.linalg.vector_norm(tangent).item() < 1 - eps:
            raise RuntimeError("read-only identity block contradicted")
        torch.testing.assert_close(output[176:], state[176:],
                                   rtol=0, atol=0)
        cases += 2

        # Exactly a no-interference orthogonal write: all subsequent writes to
        # key e2 should merely multiply the original e1 memory by alpha.
        # Projection radius never activates; generated source-operation test.
        alpha=torch.tensor([[.9999]], dtype=dtype)
        beta=torch.tensor([[.5]], dtype=dtype)
        writable=torch.ones((1,1), dtype=torch.bool)
        key=torch.zeros(1,8,dtype=dtype);key[:,1]=1
        zero=torch.zeros(1,8,dtype=dtype)
        store=torch.zeros(1,8,8,dtype=dtype);store[0,0,0]=1
        for _ in range(8192):
            store=source.gated_write(store,key,zero,alpha,beta,writable)
        target=.9999**8192
        if abs(store[0,0,0].item()-target) > (2e-4 if dtype==torch.float32 else 1e-8):
            raise RuntimeError("orthogonal source-writes did not exhibit compulsory decay")
        cases += 1

        # Orthogonal keys are not sufficient for exact old-read preservation
        # when an otherwise admissible unit-value write triggers global
        # Frobenius projection. The old q=e1 read becomes smaller even after
        # accounting for the explicit alpha decay. This is an actual source
        # counterexample to overpromising "orthogonal write preserves facts".
        saturated=torch.zeros(1,8,8,dtype=dtype)
        saturated[0,0,0]=4.
        orthogonal=torch.zeros(1,8,dtype=dtype);orthogonal[0,1]=1.
        unit_value=torch.zeros(1,8,dtype=dtype);unit_value[0,0]=1.
        keep=torch.full((1,1),.9999,dtype=dtype)
        # beta=0.5 is attained by a finite sigmoid logit of zero.
        strength=torch.full((1,1),.5,dtype=dtype)
        old_dropped=source.gated_write(
            saturated,orthogonal,unit_value,keep,strength,writable)
        if not (old_dropped[0,0,0] < (4*keep[0,0]) and
                torch.linalg.vector_norm(old_dropped.reshape(1,-1),dim=-1).item() <= 4+1e-6):
            raise RuntimeError("Frobenius projection did not couple orthogonal reads")
        cases += 1

        # An old memory read differs even on a new unit key after strong
        # overlap. A finite matrix has no universal zero-interference theorem.
        old=torch.zeros(1,8,8,dtype=dtype)
        old[0,0,0]=1
        key=F.normalize(torch.tensor([[1.,1.,0,0,0,0,0,0]],dtype=dtype),dim=-1)
        value=torch.zeros(1,8,dtype=dtype)
        updated=source.gated_write(old,key,value,
                                  torch.ones(1,1,dtype=dtype),
                                  torch.ones(1,1,dtype=dtype),writable)
        if not updated[0,0,0].item() < 1:
            raise RuntimeError("nonorthogonal read interference unexpectedly absent")
        cases += 1

    print({"status":"PASS_NATIVE_BOUNDED_FORWARD_AND_NONCONTRACTION_DIAGNOSTICS",
           "source_commit":"081e534a074001c6a3b2251b3ff31ad20ec4fca8",
           "checked_transition_cases":cases,
           "highest_observed_associative_Frobenius_norm":peak,
           "fast_slow_max_abs_bound":1.0,
           "memory_Frobenius_bound":4.0,
           "strict_whole_state_contraction_ruled_out_by_read_only_memory":True,
           "trained_checkpoint_tested":False,"real_world_skill_established":False,
           "original_weights_changed":False,"training_steps":0})


if __name__=="__main__":
    verify()
