# LANGBLEND032 — no-fit exact-checkpoint TAC / FIT-only P3 probability-mixture results

**2026-10-28 · internal exposed DEV diagnostic, no new model fits or optimizer steps.**

Original [GitHub Actions run #40782273000](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40782273000) **PASS**. Source-run artifact: [12857971651](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40782273000/artifacts/12857971651). Result JSON SHA256 **cb07440ac966c8884ebca836059329fffa59cd22dcfbf85dcc30e03993050748**. Preregistered in draft PR #560.

The experiment:
- Loaded **all five original** LANGCOMPARE031 320-update, 500,122p TAC-CONTEXT026 original trained checkpoints from the original GH artifacts, verifying SHA256 and the original source-bound sample-order digest. No altered weights.
- Constructed the exactly source-verified original FIT-only P3 model (Dirichlet alpha 0.5, recursive k=1..3, tau 8) from 13 genuine publisher FIT documents and the authentic Unicode FIT supplement.
- Used the unchanged frozen four **previously exposed** PSF CPython tutorial DEV files, 4×8×128 next-byte targets = 4,096 per seed. P3 sees only up-to-time-t 3-byte prefixes, TAC sees only up-to-t inputs. No future-byte, cross-document source ownership, protected TEST, GPU work or model retraining.
- Checked **lambda 0** agrees with original five-seed TAC NLL at 1e-5 tolerance; **lambda 1** agrees with the independent registered original full P3 scoring function at 1e-8 tolerance. No DEV-selected lambda and no dropped seeds.

## Entire preregistered probability-level mixture grid

p_lambda(y|h)=(1-lambda) p_TAC(y|h)+lambda p_P3(y|h). These are actual probability mixtures; not logit averaging, not a new native model, and not an additional trained module.

| P3 probability weight | Five-seed mean NLL ↓ | Mean top1 ↑ |
|---:|---:|---:|
| 0.00 (TAC only) | 2.099715 | 43.81% |
| 0.10 | 1.988873 | 45.52% |
| 0.25 | 1.899796 | 48.13% |
| 0.50 | 1.814500 | 51.21% |
| **0.75** | **1.777898** | **51.93%** |
| 0.90 | 1.786360 | 51.68% |
| 0.95 | 1.802048 | 51.63% |
| 0.99 | 1.834781 | 51.66% |
| 1.00 (P3 only) | 1.877306 | 51.71% |

0.75 was one point on the *frozen* grid and gave the lowest **observed exposed-DEV** mean NLL; it was not chosen/tuned ahead of evaluating a new independent holdout. At 0.75, mean NLL 1.777898, SD 0.007804, mean Brier 0.615132, mean top1 51.93%, 2.564964 bits per original byte. Improvement over P3-only on these exposed targets: **0.099408 NLL (5.295% relative)**. This is a **HYBRID INFERENCE GAIN**, not evidence that native TAC has beaten P3.

## Five original TAC seeds; P3 baseline fixed across seeds

| Seed | Native TAC λ0 NLL | TAC+P3 λ0.75 NLL | P3-alone λ1 NLL |
|---|---:|---:|---:|
| 1701 | 2.123229 | 1.780554 | 1.877306 |
| 1702 | 2.070750 | 1.769542 | 1.877306 |
| 1703 | 2.095374 | 1.773546 | 1.877306 |
| 1704 | 2.108064 | 1.775950 | 1.877306 |
| 1705 | 2.101159 | 1.789897 | 1.877306 |

**All 5/5 seeds** show a 0.75 hybrid NLL improvement over P3 under this frozen exposed-DEV protocol. This confirms complementarity for the current narrow distribution, not intrinsic native memory superiority. Because all five seeds share the same four DEV documents and P3 counts, these are not five independent source tests.

### Scientific limitations and prioritized next experiment

- DEV was **already exposed** in architecture iteration and so these results are internal exploratory evaluation. A new publisher-licensed, disjoint, genuinely unseen test source must be frozen prospectively before selecting lambda for real-world claims.
- Full combined model costs include P3's FIT-only count table and lookup operations, not merely TAC's 500,122 neural trainable parameters. No measured *combined* inference latency, RAM or deployment memory comparison yet.
- Neither the BDH nor Transformer receives a P3 blend in this run. The output cannot establish a hybrid-vs-hybrid architecture comparison.
- Do not convert the observed lowest-DEV lambda into a production default without new unexposed evaluation.
- For mathematical architecture improvements, prioritize a separate **native-only** content-conditioned erase/write operator and a reallocated equal-500K parameter no-erase control; measure gradient flow, key collision and prefix causality, then genuine sources and matched compute. Avoid treating a count lookup as proof of learned native long-horizon memory.

**Original failed results, original LANGCOMPARE031 15 trained checkpoints and older LANGTRAIN025 failures were not modified.**
