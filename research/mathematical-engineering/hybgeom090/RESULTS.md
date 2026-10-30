# HYBGEOM090 — executed no-fit experiment results (2026-10-30)

**Status:** NEGATIVE frozen-weight FIT diagnostic; math/engineering checks passed. No new training or qualified architecture comparison. Experimental [draft PR #638](https://github.com/FPC-effortless/TAC-OSM/pull/638), stacked on original archive [draft PR #637](https://github.com/FPC-effortless/TAC-OSM/pull/637).

## Original execution evidence

- [Successful GitHub Actions run 41005001634](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41005001634), original source head `58f40b97186cd0672facd5b09490218437540de0`.
- [Original machine-readable PLAN / PREFLIGHT / RESULTS artifact, ID 12955490734](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41005001634/artifacts/12955490734) (90-day retention). The JSON contains full six seed/document cases, original checkpoint and publisher owner hashes, six-arm NLL/accuracy, CPU process-time receipts and unknowns.
- 34 scoped pytest checks **PASS**, including actual original TAC source integration, frozen original archive integrity and research-probe contracts. No reported failures or skips.
- Phone lab: `READY_INTERNAL_RESEARCH_PLAN`; parity preflight: `PASS_DIAGNOSTIC_ONLY`; no architecture-superiority eligibility.
- Source: two original CPython publisher FIT owners (`Doc/howto/unicode.rst`, `Doc/tutorial/whatnow.rst`), original FIT source digest `911981cc63fd5436e0495cb067784e53475971131c811b7026e07d97a886d2d4`. Each original full streaming case used 320 observed bytes, with 256 scored next-byte targets (ordinals 64–319), three original independently trained 1,280-step seeds 1701/1702/1703: **1,536 paired targets**. All original checkpoint archive/member SHA256 and ancestry verified.
- Exact original neural parameters per seed **100,521**, **unchanged and frozen**. New fit optimizer steps 0, GPU 0 minutes, DEV scoring false, TEST access false.

## Pre-registered frozen no-fit six-arm comparisons

Mean NLL lower is better. The `delta NLL` column subtracts the untouched original. Accuracy is next-byte top-1 measured on the **same** source FIT rows. All variants use frozen exploratory deterministic corrections rather than equally trained architectures.

| Condition | Mean NLL ↓ | ΔNLL vs original | Accuracy ↑ | Cases with lower NLL than original (of 6) |
|---|---:|---:|---:|---:|
| Original untouched TAC | **1.779355** | — | **50.3255%** | — |
| Frozen fixed Euclidean | 1.784595 | **+0.005240** | 50.1302% | 2 |
| Frozen contextual Euclidean | 1.784553 | **+0.005198** | 50.0651% | 2 |
| Frozen contextual SPD metric | 1.784587 | **+0.005232** | 50.0651% | 2 |
| Hybrid continuous/event dynamics | 1.794747 | **+0.015393** | 49.8698% | **0** |
| Hybrid + contextual SPD metric | 1.800578 | **+0.021224** | 49.5443% | **0** |

This is a **negative result**: all five modifications worsen aggregate frozen FIT NLL, and the two hybrid conditions worsen all six independent seed/document-case NLL differences. Because two source owners are repeated across seeds, 6 cases are **not** 6 independent documents; do not treat this as a formal uncertainty interval or population significance test. No model was trained with the new computation, so these results do not prove that a properly trained hybrid architecture would fail.

The original hybrid CPU process time averaged 0.376623 s per 320-byte streaming case, versus original 0.366137 s on the same jobs (~2.9% slower) with a different sidecar computation path. Treat this only as a small exploratory in-run CPU overhead, **not** a stable device decode benchmark. Geometry-4-arms elapsed/process receipt represents a combined correction evaluation, not an independent fully equivalent neural inference timing.

## Other falsifiers and scope

| Claim from proposal | Verified evidence | Research verdict |
|---|---|---|
| Exact smooth flow, bounded disturbance | Local exact exponential linear `dx/dt=-λ_q x+w` with chosen positive rates, shape, finite/zero-time and disturbance tests | **PASS mathematical fixture**, not a proof for entire TAC |
| Discrete transitions and nonexpansive resets | Deterministic reset / transition / clock and dwell validation tests | **PASS constructed contract**, arbitrary learned triggers and all TAC resets OPEN |
| Contextual Euclidean representation | Context changes computed distances, shapes and output correction connection | **PASS functional mechanism**, human semantic quality UNKNOWN |
| Context-dependent SPD tensor metric | Positive-definite rank-one tensor, gradient/finiteness, straight-line length checks | **PASS local mechanism**; global geodesic and learned metric gains OPEN |
| Robustness under disturbances | Selected isolated flow obeys energy inequality and reset factor <=1 by construction | **PASS local flow only**; full-model bounded-state/accuracy tradeoff OPEN |
| Adaptation and regret under shift | Constructed toy example: fast adjustment improves one shift but harms no-shift | **MIXED constructed control**, no real agent task changes or regret claim |
| Micro-macro closure | Constructed linear mean closure true; equal-mean nonlinear populations yield different next mean | **Counterexample to naive sufficiency**, no real multi-agent scaling test |
| Future leakage / original trained graph / source owners | Scoped source tests, original archive hashes and prequential next byte scoring | **PASS scoped tests**; not a general information-flow proof |
| Genuine semantic judgments, real elapsed time, new trained model, head-to-head resource-matched arms, PNDS Lean 4 and Python full computational graph | Not supplied/executed | **OPEN / NOT TESTED** |

**Toy example detail:** For an artificial 16-step target switch, cumulative square error for a slow vs fast control was 7.58863 vs 0.58198; with unchanged target it was 2.10787 vs 13.50002. These are scripted fixtures, not neural agent outcomes. Mean-closure nonlinear fixture with population states `[1,-1,1,-1]` versus `[0,0,0,0]` shows the same mean 0 and distinct next means 0.2 vs 0.

No human semantic benchmark, no physical timestamps, and no actual multi-agent data were provided or authorized by the current frozen CPython FIT protocol. Byte ordinal is **not** elapsed time. Do not claim any geometric intelligence, adaptive cognition or full TAC-OSM Lyapunov stability from the demonstrations. No trained weights or source datasets were overwritten, no merges or independent certification.

## Next eligible study (not yet run)

Keep this as a **draft negative result**. To scientifically decide the user hypotheses, register actual real interaction timestamps, independently labeled held-out context-dependent semantic judgments, adaptation changes and sources, and real multiple-agent state trajectories with explicit rewards/costs where present. Include a simple discrete recurrent core, hybrid core, contextual Euclidean and SPD variants **trained from scratch** under exact current program matched actual parameter (<1% gap, target 95k–105k unless owner formally changes the budget), FIT exposure and optimizer/FLOPs; 5 seeds, independent-case uncertainty, source-time anti-leakage, archived original checkpoints and cleaned replay, latency and boundedness reports. Full PNDS Lean/Python source-graph mathematical record must be independently developed and verified before any formal math qualification. Protected TEST remains sealed.
