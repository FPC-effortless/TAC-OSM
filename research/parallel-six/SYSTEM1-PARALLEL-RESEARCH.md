# TAC System 1 — source-grounded experimental neural contract

2026-10-26. **Untrained neural candidate, not verified intelligence, latency, calibrated risk, or production control.** Isolated implementation based on six-family parallel consolidation draft PR #390. PNDS remains the mathematical foundation, not an additional model family.

## Scope and tensor graph

Let authentic observed state x be [B,T,14] and the *logged previously executed action* u be [B,T,14]. Existing source-clock, identity, action timing and provenance admission checks must be enforced outside this tensor core. System1ParallelWorld reuses the actual ParallelTACWorld shared encoder, independent-input provisional fast/slow proposal, norm-bounded 8×8 right-affine associative memory, read-before-write retrieval, and read-refined fast/slow features. The existing world_stream_step uses the **same** tensors, weights and memory schedule, not a second fitted network.

The inherited prediction is xhat[t+1] = x[t] + delta_scale * tanh(world_readout(f[t],s[t],r[t],e[t])), where delta_scale must be estimated only on fitting observations. The new head consumes h[t] = concat(f[t],s[t],r[t],xhat[t+1]) (dimension 2*88+8+14 = 198) and computes:

- z = SiLU(W1*h+b1), W1 shape [12,198].
- (ell, q) = W2*z+b2, W2 shape [15,12]. Scalar ell is an **uncalibrated** adverse-next-event logit; q is 14-dimensional dispersion.
- adverse_event_score = sigmoid(ell), residual_scale = softplus(q) + 1e-4 > 0.

The head adds **2,583** trainable parameters to the existing 93,324-parameter shared graph, for **95,907** total at these exact dimensions. There is no separately trained family memory engine. Output is observational advisory only; scores are NOT calibrated probabilities, causal action utility or permission for ALLOW/ABORT. Actual case identity and as-known-at authority are external typed input responsibilities.

## Computation and memory

The original causal batch path keeps input-parallel preprocessing and a chronology-preserving associative scan with O(T) 8×8 work and O(log T) scan depth; the new head is time-parallel. Streaming carries the existing 88-dimensional running basis sum and 8×8 matrix = 152 FP32 floats = 608 bytes per entity, **excluding** weights, identifiers, timestamps, runtime buffers and event count. Each streaming step has computation independent of history length; this is not a measured low-latency SLA. The head increases compute and must be benchmarked fairly.

## Mathematical assertions and proof status

- The source prefix uses exact-real right-affine composition: M_next = M @ P + Q with externally schedulable input-only gates. This does not prove numerically identical IEEE reductions or long-range retrieval usefulness.
- Over exact reals, softplus(q)+epsilon > 0 and 0 < sigmoid(ell) < 1. With finite-precision extremes, sigmoid can saturate to an endpoint.
- Causal, stream/parallel, floating-state bound and autograd checks are only software engineering checks. No complete Lean proof or full finite-precision or AD refinement of the new head is claimed. PNDS must receive source-linked graph/activation/rounding obligations.
- Joint gradients of reconstruction, point, uncertainty, risk and future decision losses may conflict. No scientifically endorsed weighting or optimizer intervention is implied by unit tests (see DIAGGRAD001).

## Falsifiable research gate

1. **Authentic labels first.** Choose a real, rights-cleared dataset with timestamped observed states, logged already-executed actions, measured next states and genuine adverse/success outcome labels. ALOHA action-to-joint trajectories alone are not ground truth for hazardous outcomes; never manufacture risk labels to claim real-world skill.
2. Register actor/entity/site and chronological FIT/DEV/TEST splits, label observation horizons, missingness and recording delays before tuning. Future actions, next states, backdated revisions or protected held-out outcomes must never enter causal inputs or normalizers.
3. Fit the full neural graph and matched GRU, SSM/Mamba-compatible, attention and native TAC controls, with exactly matched source splits, exposures and training/tuning FLOPs. Active/deployed parameters target 95k–105k with <=1% comparator mismatch. Include persistence, velocity and servo physical references as applicable.
4. Report next-state error, proper uncertainty score and coverage; adverse-label AUROC, AUPRC, Brier/ECE and abstention **only if labels exist**; delayed-fact/entity interference, no-lookahead, ablated associative read, runtime p50/p95 latency, peak weights and state, energy and identity leakage. State real sensor/robot units.
5. Qualification requires independent untouched held-out outcome evidence, multiseed paired intervals, non-regression on high-cost events and decision cost, and source-linked PNDS reasoning. No safety or realtime certification follows from CPU tests.

No training, original checkpoints, original experiment records or past branches were modified. Tests with constructed numeric inputs cover implementation invariants only.
