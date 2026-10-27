# GP-UNC-001 — engineering diagnostic results and qualification gaps

**Date:** 2026-10-27. **Evidence class:** constructed numerical and source-time software verification only. No new model FIT, training authorization, independent authentic held-out TEST or new GPU run.

## Source-pinned implementation

- Exact small dense RBF GP Gaussian-conditioning reference: `src/tac_osm/gp_uncertainty_nofit.py` (maximum 128 reference training samples).
- Projected-process inducing-reference filter: `src/tac_osm/gp_sparse_nofit.py` (maximum 64 inducing coordinates; fixed numeric state; source-time guarded residual update). This is a *low-rank Gaussian approximation*, not a variationally optimized sparse GP, trained neural residual head, or calibrated decision policy.
- Class stores the symmetric precision matrix, information vector, Cholesky basis, and inducing vectors. Retained state grows in inducing count and feature dimension but **not number of observed source events**. Each update currently recomputes the small Cholesky to fail closed and is **O(m³)** worst-case arithmetic; do not claim measured ultra-low step latency or O(m²) update time. Online model parameter counting and hardware latency remain OPEN.
- Six-family × five-arm registration: 30 cells `NOT_FITTED_OR_QUALIFIED` for System 1, Enterprise, Temporal, Physical, Edge-Lite, Counterfactual.

## Executed verifications

| Evidence | Outcome | Scope |
|---|---|---|
| Small exact GP no-fit properties | 11/11 passed | Constructed covariance, NLL, conformal radius, source-time checks |
| Independent NumPy FP64 posterior comparison | 72/72 constructed configurations passed | Max mean absolute difference ~1.15e-13; variance ~1.11e-15, tolerance 1e-10 |
| Inducing projected filter | 10/10 passed | Constructed on-basis vs exact equality, off-basis discrepancy, constant-sized state, chronological rejection, invalid inputs |
| Dedicated TAC GP mathematical workflow | PASS | GitHub Actions run 40479488795; no training |
| Full TAC CI at scientific test-count correction | 1190 passed | GitHub Actions run 40478806536; at frozen code pin 21a5ffef; subsequent CI must also pass |
| PNDS Lean theorem build and Python reference | PASS on PR #45 initial CI | `PNDS/GPUncertainty.lean` scalar bounds and `tests/test_gp_uncertainty_math.py` constructed input contracts; no full-model proof |

**Numerical nuance:** the 72 checks are configurations inside a separate unit test, not 72 independent research trials or trained seeds. The 10 inducing-filter checks are additional constructed diagnostic tests. One-step conditional uncertainty is not equivalent to a calibrated real-world confidence guarantee.

## Negative and open scientific results

- The projected approximation is **not** equal to dense exact GP in general; test deliberately confirms off-basis mismatch. At observed inducing coordinates with the same Gaussian noise model, the tiny reference matches exact GP within the registered numerical tolerance.
- Original real-source LATFUSE019 rainfall-tail NLL and SO2/wind undercoverage remain unresolved; none of this study changes those original outcomes.
- Neural TAC Gaussian variance head, residual sparse GP on actual trained TAC latent states, conformal calibration on an independent legally sourced chronology, matched GRU and real cohort tasks for **each** six-family path: NOT EXECUTED.
- Full PNDS multivariate proof, source-equivalent TAC graph, finite-precision and automatic differentiation correspondence, authenticated real source/labels, 95k–105k useful parameter verification, matched training compute, independent signed source/outcome/scorer/ACL authorization, sealed TEST evaluation, checkpoint replay and causal Counterfactual identification: **OPEN**.
- No claim of prediction quality, probability calibration, memory benefit, decision utility, source provenance, policy safety or commercial readiness follows.

Research follow-through: [TAC PR #491](https://github.com/FPC-effortless/TAC-OSM/pull/491), [PNDS PR #45](https://github.com/FPC-effortless/pnds-lean/pull/45), [GP study issue #492](https://github.com/FPC-effortless/TAC-OSM/issues/492). Preserve historical results; both proposed PRs are draft and unmerged.
