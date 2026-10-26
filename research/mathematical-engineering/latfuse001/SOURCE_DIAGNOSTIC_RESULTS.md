# LATFUSE001 SOURCE-ONLY-DIAG-001 — first genuine FIT-source, zero-fit result

**Date:** 2026-10-26
**CI run:** https://github.com/FPC-effortless/TAC-OSM/actions/runs/40413374694 — completed SUCCESS
**Registration:** \`research/mathematical-engineering/latfuse001/FAIR_COMPARISON.json\`, PASS_DIAGNOSTIC_ONLY
**Trained model count:** ZERO
**Scientific capability/architecture superiority:** NOT TESTED / NOT ELIGIBLE

## Source access, provenance and chronology
Exact original UCI501 Beijing Multi-Site Air Quality CC-BY-4.0 archive SHA256:
\`d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8\`.
Retrieved verbatim from pre-existing original experimental commit
\`6e01290ddf3f10d298c4bcf463a219c9728c0565\`, no original artifact modified.
No 2015/2016/2017 numeric records parsed. Only first512 continuous March 2013 hourly observed rows ×12 real source stations; earliest 256 used for input-only per-channel normalization and subsequent 256 to inspect forward recurrent states. 13 sensor channels, 39,760 finite and 176 missing entries in diagnostic phase. No data labels or target outcomes scored, no 2015 reporting, no new split of protected evaluation sets. Untrained random parameters, seed20261026.

## Registered measured initial-state values
| Quantity | Observation |
|---|---:|
| Model active/deployed learned parameter count | 20,087 / 20,087 |
| Mandatory trained candidate range | 95,000–105,000 (**FAIL for training**) |
| Batch/stream source station IDs | 12 authorized source sites |
| Recurrent float values per stream | 832 |
| Nominal FP32 carried state bytes incl16 bool flags | 3,344 |
| Numerical diagnostic dtype | FP64 (reporting FP32 storage size separately) |
| Mean retention gate alpha | 0.2778576 (min0.1801642, max0.4147684) |
| Mean write gate beta | 0.7005743 (min0.5528515, max0.7860517) |
| Fast latent norm mean | 4.0972029 |
| Slow latent norm mean | 1.5920749 |
| Associative latent norm mean | 1.3835152 |
| Occupied station slots | one per independent station stream |
| Training updates | 0 |

The alpha statistic is conditional on the **randomly initialized network** and these real input values. It is neither a trained-value estimate nor a direct guarantee of surviving memory under ongoing writes: delta corrections can also change old association directions. The explicitly *retention-only* multiplier using mean alpha becomes negligible by64 updates; the diagnostic JSON rounds it to zero. The slow scalar state has direct homogeneous sensitivity \`0.99^H\`: at H512 only ~0.583% remains, conditional on fixed future inputs. This does not establish full-model catastrophic forgetting.

## Constructed stress diagnostic (NOT original real-world outcome accuracy)
Normalized independent random keys with d_k=16; unit beta and unit retention. Relative L2 retrieval error:
| N keys | Relative L2 error |
|---:|---:|
| 8 | 0.411517 |
| 16 | 0.701259 |
| 32 | 0.994925 |
| 64 | 1.235607 |

These are seeded no-learning operator tests, not "8 keys max" or proof of model ability. They expose crosstalk even below the ambient key dimensionality for the selected random-key, sequential delta-rule setting.

Single-session CPU reference calculation of 64-step signed affine state sequence (8×32): serial0.3833ms, Hillis–Steele prefix0.4340ms, exact final difference recorded0.0. **Do not extrapolate these timings to trained batch throughput, GPU kernels, or inference superiority**: overhead/measurement noise/hardware differ, prefix is unoptimized and only that subgraph can scan.

## Engineering diagnosis and design candidates (hypotheses only)
1. **Initialization retention likely too aggressive:** sigmoid retention has no initial near-unity prior and measured 0.278 on real FIT inputs. Experiment with explicitly initialized high-retention prior and inspect trained effects under matched compute and leak prevention. Do not optimize on protected results.
2. **Memory addressing interference:** normalized keys and gated delta do not guarantee orthogonal retrieval or entity independence in the shared matrix. Compare key decorrelation, adaptive grouped keys and isolated source slots, measuring effective key rank and long-lag retrieval.
3. **Slow-memory temporal limit:** fixed0.99 decay offers insufficient direct one-time memory survival at H512; consider multiple bank rates and learned source-admission, without claiming nonlinear recurrent parts can be parallelized.
4. **Model too small for research training:** 20,087 parameters. Engineering a useful ~100k parameter active model, original real-data adapter, safe logged candidate semantics and unambiguous forecast targets must precede any optimizer update.
5. **Mobile cost:** current 3,344B per independent stream is ~5.5× historical ME110 608B (nominal only), and reference scan's CPU elapsed is not proof of low latency. Test realistic separate tiny Edge-Lite settings rather than marketing this 16-slot candidate as mobile-grade.
6. **Scientific missingness:** no trained checkpoint, no real outcome predictive scores, no native-matched equivalent model/compute, no five-seed run, no independent replay or protected TEST. Current data can support diagnosing input paths only.

## Next experiment gate
**Do not promote or fit LATFUSE001 yet.** First conduct model source design review: target matching 95k–105k real trainable parameters; data cutoff and legitimate future-known conditioning (if air quality: calendar, not synthetic "action"); source-authorized station IDs; forecast shape/mask/time; independent audit; preregistered original real cohort/splits/outcome/seed budget; repository-owned checkpoints before any TEST; fair matched baseline if claiming superiority. Continue separately with original native architecture diagnosis after this candidate's capability test is either qualified or honestly blocked.

All historical ME110 FAIL0/5 results and active ME111 drafts remain unchanged. No code, source, checkpoint or negative findings were overwritten.


## Extension: H80/96,199-parameter genuine FIT forward/backward preflight (2026-10-26)

**Pass:** https://github.com/FPC-effortless/TAC-OSM/actions/runs/40414352512 (source no-fit CI, exact commit f85da1b324734d87533129742284dec3f3dfa06f).
**Earlier failed attempt, preserved rather than hidden:** https://github.com/FPC-effortless/TAC-OSM/actions/runs/40414207796 failed only because the prefit script passed a string path to the source reader's Path.read_bytes method. No training ran; the exact parser-type defect was fixed and rerun PASS. The numerical source/research protocol did not change.

The same untrained LATFUSE001 graph, configured as H80, key16, value16, 16 entity slots, 13 original source input channels and 6 **future-known calendar** (not do(action)) covariates, produced:

| Real-source no-fit preflight | Result |
|---|---|
| Genuine source SHA256 | original UCI501 archive, independently checked |
| Source time | 2013 March, historical input hours256..319, source cutoff319 |
| Legal future calendar conditioning | deterministic 2013 hour325 (+6), not intervention |
| Future targets | 156 actually observed FIT-only channels accessed **after** forecast |
| Instantiated trainable/deployed parameters | **96,199/96,199** |
| H80 carried state (16 slots, 16×16 associative) | **1,696 floats +16 flags =6,800 nominal FP32 bytes/stream** |
| Absolute 100k target 95k–105k | **PASS (size only)** |
| Active autograd tensors | **32/32**, all finite measured source gradients |
| Weights before/after probe | **bit-exact unchanged**, no optimizer |
| Read-only candidate forecasts independent of actual future labels | **PASS (scoped source call order only)** |
| Actual training updates | **0** |
| Independently reviewed full-source anti-cheat | **INCONCLUSIVE / NOT DONE** |
| Actual future-outcome capability / calibration | **NOT MEASURED** |
| Fair matched-architecture comparative parity | **NOT REGISTERED** |
| Checkpoint publication and independent replay | **NOT AVAILABLE (no fits)** |
| Entire-model PNDS/AD/IEEE/probability-law refinement | **OPEN** |

This establishes source-grounded **trainability** of the widened graph, not scientific performance. It does not retroactively qualify default H32, nor permit model training merely because the parameter count passes. Any actual fit needs a prospectively source-controlled and independently audited source-specific training protocol, all required receipts, protected eval seal, checkpoint publication/replay strategy, and an appropriate fair comparator for architecture-superiority claims. The 6D exogenous calendar is *not* action-conditioned causal control; do not market forecasts as identified counterfactuals.

**User-defined sequencing:** LATFUSE001 remains a separate candidate until a genuinely qualified predictive test or an explicit blocker conclusion. Return to the **original native architecture diagnosis** after candidate evaluation, tracked by https://github.com/FPC-effortless/TAC-OSM/issues/444. Do not conflate LATFUSE001 randomized memory weaknesses with trained native ME110 failures.
