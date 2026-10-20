# ME-011F — First frozen authentic transformer future and cross-site failure

**Original prospective first-exposure source SHA:** `ec6d34ac0cc79e956eb492f8b4c03096058049b9`.
**Successful independent GitHub Actions [run 39388444587](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39388444587)**. Full original scoring artifact `12248810965` (`me011f-original-real-etth1-etth2-future`), original JSON SHA-256 `149868f2abeaa7b68274468746be0541748872054d5de890201bc81f8b3a9848`. Native and GRU weights from original training [run 39388044790](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39388044790), artifact `12248760583`, source commit `957d45e939e66d8f0bbce8d1f7537bed6e2f6d69`. **All six checkpoint SHA-256 digests checked before new source readings**; zero test optimizer steps.

## Authentic source, held-out windows
- ETTh1: same transformer but future chronological rows [11640,14520), official ETTh1 SHA256 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`; 452 sampled evaluation forecast windows.
- ETTh2: **different physical transformer station**, also rows [11640,14520), publisher official ETTh2 MD5 `51a229a3fc13579dd939364fefe9c7ab`, newly recorded SHA256 `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b`; 452 independently scored forecast windows.
- For each real record: six electrical-load channels and one oil-temperature channel, genuine hourly chronology, 96 previous time steps, horizons 24h/168h; same ETTh1 TRAIN-only normalization applied to all data including cross-site. No future labels fed as model inputs. Neither score was used to refit or select model weights. Source rows >=14520 were not parsed by evaluation.
- Exactly 6 original model checkpoints (3 TAC, 3 GRU), authentic targets, zero quantile crossing. Untrained baselines last-value persistence and trailing-24h observed mean, both with degenerate intervals **NOT treated as calibrated probabilistic benchmarks**.

## Primary untouched test results — median oil temperature absolute error in physical °C, lower better

| Real physical site | Persistence | 24h trailing mean | Native TAC (3 seeds) | GRU (3 seeds) |
|---|---:|---:|---:|---:|
| **ETTh1 future chronological** | **1.97198** | 2.00422 | 4.86245 | 4.46514 |
| **ETTh2 unseen station** | **3.79078** | 4.45806 | 4.19908 | 4.47618 |

- ETTh1 FUTURE: TAC **2.47× worse** than simple persistence; GRU **2.26× worse**. GRU better than native TAC mean.
- ETTh2 CROSS-SITE: native TAC better than GRU mean (4.199 vs 4.476°C), but still **10.77% worse** than persistence (3.791°C).
- Individual native seeds, ETTh1 future: **5.40018, 4.10332, 5.08385°C**; GRU: **4.43992, 4.52478, 4.43072°C**. No claim that any single seed demonstrates general model superiority.
- Individual native seeds, cross-site ETTh2: **4.22174, 3.92956, 4.44595°C**; GRU: **4.39785, 4.61681, 4.41388°C**.

## Proper quantile calibration and interval sharpness

| Site | Native central 80% empirical coverage | GRU central 80% coverage | Native 3-quantile pinball | GRU pinball |
|---|---:|---:|---:|---:|
| ETTh1 future | **18.88%** | 34.29% | 0.20458 | 0.16767 |
| ETTh2 cross-site | **40.12%** | 36.58% | 0.17289 | 0.19008 |

Nominal target is **80%**. Predictive intervals severely under-cover; a mathematical noncrossing construction gives **no coverage guarantee**. Native predicted central interval widths averaged **5.95°C** (ETTh1) and **5.52°C** (ETTh2); GRU **7.16°C** (ETTh1) and **5.36°C** (ETTh2). This suggests both location shift and uncertainty miscalibration, but mechanism attribution needs additional preregistered diagnostic evidence.

## Scientific decision

**FAIL** the preregistered gate requiring superiority over GRU AND persistence on both stations and reasonable predictive-interval calibration. The native recurrence is mathematically bounded, but that says nothing about correctness on a shifted physical domain. Retain all original favorable ME004–ME010 MetroPT results AND current negative ETT evidence without silent model/branch selection.

No native TAC or GRU predictive state here can be promoted for autonomous operational decisions, real safety or commercial deployment. ETT does not furnish verified interventions and outcomes; it cannot qualify neural CASM decision utility or PNDS receipt validity.

**Next mathematical priority:** distribution-free **split conformal quantile calibration** and strict causal delayed-feedback online calibration, with explicit coverage assumptions. First test on *later unused rows 14520 onward*, using earlier historical dev data exclusively for calibrating frozen models; never tune on opened [11640,14520) future results. Compare calibration widths, coverage, and MAE, and investigate whether both models are systematically biased under seasonal drift. Strong alternative: calibrated persistence/seasonal-naive quantile baselines and independent more recent operational data.

TAC-OSM remains its own native learned predictive world-state architecture **not an SSM**. All unified plan ideas including CDL-PHR, BDH-like local plasticity, self-organizing sparse entity dynamics, executable operators, adaptive reasoning and PNDS/Lean safety proofs remain recorded as research questions, with honest separate evidence gates. No automatic merge.
