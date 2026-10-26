# ME108 neural mathematical/finite-precision/autodiff/probability-law refinement

2026-10-26. **Source code/formal proof improvement, NOT completed whole-model proof or new trained capability.** Five original role-specialized ME108 102,713-parameter trained checkpoints and all FAIL0/5 original real-world qualification outcomes are preserved verbatim. This research branch adds only read-only checking and proof documentation, no source model parameter, optimizer, graph, loss/gate, architecture or protected target changes.

## Formal source target, truth in advertising

The exact-real full finite-layer source-shaped ME108 model already appears in PNDS/SensorGraph105.lean, RoleSpecializedFFN.lean and RoleGuardedTraining.lean. Those are typed formulas and limited mathematical lemmas, not total correctness for the actual CPU/CUDA PyTorch kernels.

The companion [PNDS draft PR23](https://github.com/FPC-effortless/pnds-lean/pull/23) implements:
1. Source-shaped 102→16→12/12/6 FFN input JVP/VJP duality via finite-sum algebra, weight/bias differentiation algebra and the exact scalar tanh derivative; no opaque admitted theorem.
2. A conditional scalar affine roundoff error bound, *assuming* local rounded operation errors; this is NOT an IEEE-754 bit-level implementation proof or whole-model error accumulation bound.
3. Measurable asymmetric split-to-raw and zero-censored probability-law pushforward: normalized total mass one and no mass on negative recorded values, including an instantiation of Mathlib's true standard Gaussian.
4. An explicit ledger of still missing source-certified full AD (every operator/objective, masks/subgradients/detach), IEEE FP32/64+transcendental kernels and reduction order, equivalence between Gaussian-measure law and native ndtr/log_ndtr/ndtri/NLL/quantiles, optimizer semantics, and one composed source-to-output theorem.

**No formal proof of all these open propositions is claimed.** Mathematical probability normalization does not establish empirical90% coverage (ME108 ordinary88.67%, one gap98.60%) or neural predictive qualification (ME108 zero of five full gates).

## Read-only original ME108 source-check experiment

[GitHub Actions run40268747906](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40268747906) completed SUCCESS. Original artifact**12626708917** stores original per-checkpoint per-role precision results. New script scripts/me108_neural_refinement_audit.py:

- Fetches only authentic original FIT2013–2014 UCI501 rows (eight recorded-origin source histories), never opens protected 2016–2017 target rows.
- For each of the FIVE original trained ME108 checkpoints108001–108005, asserts the original immutable SHA256 and 4096 optimizer updates and restores fitted model weights without performing any optimizer update.
- In BOTH FP32 and FP64, checks each of THREE 102→16→12/12/6 role FFN heads against an independent NumPy dense/tanh forward, including source feature gradient and all four weight/bias gradients. **5×2×3=30 role-level full VJP checks** in source units, inheriting unchanged2e-5/2e-6 FP32 and1e-9/1e-10 FP64 comparison tolerances.
- Checks actual censored negative CDF support, zero-atom CDF in[0,1], real quantile shape and source median inverse-quantile boundary. These are numeric regressions; mathematical identity to underlying normalized Gaussian measure and IEEE transcendental rounding remains open.
- Outputs explicit meta-status ACTUAL_SOURCE_NUMERICAL_TEST_NOT_FORMAL_PROOF and no test-promotion/robot action/counterfactual-causal claims.

Original global ME108 full-source tensor/optimizer/gradient audit and 1059 Python tests already passed on its parent branch. This dedicated audit adds independent 30 fitted head-level gradient checks; it is NOT redundant claimed whole model proof.

## Remaining engineering completion gates

| Area | Proven here | Still unproved |
| --- | --- | --- |
| Neural forward | Typed full exact-real graph inherited; role head real adjoints | full source mapping of every operator across scan + heads, over all legal source tensors |
| Autodiff | Real role FFN exact scalar tanh/JVP–VJP algebra; 30 authentic trained-source numerical checks | full objective Jacobian/subgradients at masks, max/clamp, missingness, detach, distribution branches, source AD compiler correctness |
| Finite precision | Conditional local two-stage scalar error inequality | certified FP32/64 rounding, BLAS and transcendentals, nonfinites, reduction order, long-history overflow and composed error bound |
| Probability | Genuine Gaussian pushforward total-mass1 and nonnegative support | source CDF/density/atom and NLL/quantile equality to measure, ndtr/ndtri numerical error, empirical calibration |
| Training | Old ME108 13-block computed projection checks survive | optimizer Pytorch source fully proved and finite-step generalization theorem |
| Predictive ability | Frozen real original CHECKPOINTS/FAIL gates retained | all-five full predictive gate passing on independent source/time; real safe control |

Future work must prove exact law/bit-level refinements in the same committed math branch with source authorizations, report all failures, and NEVER silently turn test coverage into a Lean theorem. This increment keeps the established six model families and original project evidence intact.
