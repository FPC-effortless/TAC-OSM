# S1MOTION002 — bounded innovation around observed-state velocity (UNTRAINED)

2026-10-26. **Architectural engineering only. No fit, real performance claim, trained-model inheritance, new family, inference SLA, policy or completed Lean IEEE/AD proof.**

## Why this exists
Original S1MOTION001 on real source: TAC mean next-state MSE 2.49658e-5, matched GRU 1.92446e-5 and causal constant velocity 3.05908e-6. Three out of three scientific gates FAIL. Original source/checkpoints and subsequent cold replay are preserved in S1MOTION001-RESULTS.md. It would be misleading to optimize an observational System1 solely against persistence when cheap physical velocity is much better.

## Explicit input-to-output graph
The existing learned System1ParallelWorld neural core receives the current observed state x[t] and source-authorized **prior logged** action u[t-1]. With a *previous observed state* x[t-1] already known at forecast time, predict:

  b[t+1] = 2 x[t] - x[t-1]                       (deterministic causal velocity prior)
  d[t]   = delta_scale(FIT-only) * tanh(neural_readout[t])
  xhat[t+1] = b[t+1] + lambda * d[t], lambda = 0.1 fixed

Every native 88D fast / 8x8 associative scan, source key, time-prefix mean, original memory gate, read-refined slow path and the observational high-motion-proxy logit is unchanged. No new learned weights or multiple learned family engines. This only changes the **point prediction composition** and introduces the explicit real prior state; it does not magically recalibrate residual_scale or infer adverse events.

For exact real arithmetic and finite tanh input, |xhat_j-b_j| <= lambda * delta_scale_j for each output dimension. This is an engineering bound, NOT proof of better outcomes, IEEE-bitwise equivalence or certified physical safety. At lambda=0, the predictor returns exact constant velocity in source-normalized coordinates.

## Honest streaming state / complexity
The original recurrent FP32 state has 88 running fast-basis scalars plus 8x8 associative matrix = 152 floats = 608 bytes. New source-authorized previously measured 14D x[t-1] adds **14 floats (56 bytes)** => **664 bytes total floating carried state** at default width/14D; additionally an event count and history-present flag, input timestamps/provenance/normalizers, weights and runtime buffers. Each step O(1) in history; batch path retains O(T) matrix work and O(log T) scan depth plus parallel one-step history shift. No hardware latency guarantee.

First unprimed streaming frame must use b=x[t], not pretend an unavailable x[-1] exists. Explicit priming requires the caller to provide actual source-verified previous observation. This implementation does not verify that provenance on its own, so authority remains an external contract. No same-time action intervention semantics. No movement classifier performance claim; inherited risk head remains uncalibrated and not authorized to act.

## Required NEXT experiment before claims
- Must compare with a **velocity-anchored GRU**, persistence and same velocity baseline, and ideally SSM/attention and causal ablated-memory controls, with equal source observations (including x[t-1]), parameter counts within 1%, exact source/time splits, matched original FIT exposures and *measured/controlled* training FLOPs. Report unavoidable FLOP differences; never claim fully fair compute when unmatched.
- S1MOTION001 DEV 25–34 has been inspected, so avoid tuning future configurations against that DEV and presenting results as prospective unseen testing. Episodes35–49 remain closed to this program. Prefer a fresh licensed authentic robot/sensor dataset with new preregistered FIT/DEV and external cohorts; genuine adverse outcomes are needed for adverse-risk qualification. Proxy high movement != safety.
- Preserve original S1MOTION001, ME, EWA, EDGEACTION negatives. Train no model automatically from source engineering or unit fixtures. Define prospective joint-state MSE, high-motion worst-case gate, event AUCPR and calibration, real CPU+device memory accounting, independent site and full PNDS refinement before any promotion.

## Engineering status
- The new module is `src/tac_osm/variants/system1_velocity_anchored.py` and accompanying tests are fixtures for only shape, source-shift causality, exactly bounded residual, unchanged shared learned parameter count, FP32 state bytes, stream/batch parity and end-to-end gradient existence.
- **NOT** a newly trained System1 improvement. No historical checkpoint was loaded or rewritten; no original DEV/test score was recomputed by this module.
