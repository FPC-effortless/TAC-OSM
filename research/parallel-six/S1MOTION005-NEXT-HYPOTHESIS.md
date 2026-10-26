# S1MOTION005 — prospective conflict-safe, bounded auxiliary-gradient budget

**Research hypothesis only (2026-10-26). NO model fit, DEV score, new result or improvement claim.** Continue from original S1MOTION004 real ALOHA towel findings: TAC point MSE improved in every seed when auxiliary-to-backbone gradients were fully isolated, but full qualification failed because the velocity gain was <1% in seed28002 and high-motion proxy AUPRC declined >10% in seed28003. The original experiment has no qualified action policy or actual adverse-event label, and complete training FLOP equivalence vs GRU was not established.

## Motivation
Full shared multiobjective training can dominate source-world state learning: real frozen FIT original TAC memory gate's weighted auxiliary gradient norm exceeded point-gradient norm by ~276×, ~1168× and ~5886× across three seeds. Binary stop-gradient fixes the upstream interference but can deprive the representation of useful event discrimination. Original causal outcome: 3/3 point improvement, inconsistent AUPRC; independent immutable 12-model replay and 198→12→15 input-tangent mathematical companion verified. **No proof that a better point/event tradeoff exists.**

## Candidate no-new-parameter gradient operation
The ordinary shared state module returns a point loss gradient `g_p` and combined auxiliary gradient `g_a` on the SHARED encoder/memory/point parameter block only. Auxiliary-head-specific parameters receive their proper BCE/NLL gradients without projection; output time authority and observational high-motion proxy definitions remain unchanged. Before AdamW:

1. If `||g_p|| <= tiny`, set the auxiliary shared-block update to zero. If either gradient is nonfinite, FAIL CLOSED (do not silently train with contaminated source).
2. If `g_p·g_a < 0`, project away the component that would oppose the point descent direction:
   `q = g_a - ((g_p·g_a)/(||g_p||²)) * g_p`; otherwise `q=g_a`. The denominator uses the actual strictly positive norm in eligible cases. Record the preconditioner and numerical error.
3. Fixed pre-registration candidate budget `rho=0.10`, `beta_max=0.25`, numerical epsilon only for denominator of norm ratio:
   `beta = min(beta_max, rho*||g_p||/(||q||+1e-12))`;
   `g_shared = g_p + beta*q`. No parameter-count change; shared memory gate, CED, associative scan and point head unchanged.
4. Actual optimizer is AdamW. **Do not claim that positive Euclidean gradient dot product proves an actual AdamW update decreases point loss.** Log AdamW first/second-moment preconditioning, parameterwise proposed applied update direction, predicted point-loss first-order change and actual FIT-only poststep point loss before validating any mechanism. Training-FLOP cost of two gradient paths and projection MUST be counted. Compare to equally resourced controls, or explicitly disqualify fully equal-compute claims.

For exact reals, if `||g_p||>0`, the projection assures `g_p·q >= 0` (zero if the original dot was negative). Thus ideal SGD's *first-order* point directional derivative under `-eta(g_p+beta*q)` is nonpositive for `eta,beta >= 0`; this is only a local algebraic fact, **not** a global convergence theorem, finite-precision proof, AD source equivalence or AdamW step guarantee.

## New real-source test rather than tuning opened towel DEV
- Candidate new authentic source: LeRobot `aloha_static_screw_driver` (publisher license MIT); must receive its own **source-only immutable revision and Parquet checksum eligibility audit**, action clock/episode authority, and provenance scan before any fit. No simulator/synthetic claims, no re-use of S1MOTION003 tape or S1MOTION004 towel DEV to tune.
- Preregister FIT/DEV/TEST episode partitions, real labels and FIT-only normalization, strict matching within each architecture of model active parameters, source exposures, optimizer updates, full measured FLOPs and intervention overhead; original 95,907-param TAC shared/isolated models plus projected-budget candidate from identical per-seed initial weights. Include the same velocity-only and GRU controls (the GRU is separate and must not be called compute matched without full budget matching). Maintain one six-family shared neural backbone and 664B primary FP32 state excluding optimizer/weights/meta.
- Predefine positive outcome: all seeds must improve next measured state MSE by >=1% vs real velocity AND equal-or-better than both original TAC controls, no >5% measured high-motion regime harm vs velocity, no >10% drop in high-motion proxy AUPRC relative to shared, with true multiseed episode-paired intervals. An independent source/robot/task and verified real hazard or decision-cost outcomes are still mandatory for general System1/decision capability.
- Run compulsory anticheat diagnostics: all possible label-leakage pathways, protected test source access, source timestamp/revision, within-episode future controls, train/test normalization, all head/gradient block reachability, actual AdamW update angle, physical units, residual signed alignment, memory-read ablations and FP32 actual state/latency accounting.
- Do NOT launch optimizer fits before source eligibility and source-controlled frozen protocol. Historical negative gates remain immutable and all PRs remain draft/unmerged.

The immediate next useful action is an **engineering-only projected-gradient kernel with exact/finite numeric tests**, followed by a new-source preflight; it is not evidence of a functioning new model.
