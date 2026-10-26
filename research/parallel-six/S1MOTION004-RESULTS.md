# S1MOTION004 — first genuine towel experiment, registered failure with a useful controlled intervention

**Original 2026-10-26 real-data FIT and first original DEV executed; **FULL SCIENTIFIC GATE FAIL (one of three seeds met all gates)**. Preserve this result. New results never alter S1MOTION001/S1MOTION003 original failures, six family registry, PNDS or the original model. PR #421 remains DRAFT.**

## Original immutable evidence
- Genuine LeRobot ALOHA static towel 50 episodes×500 recorded frames, 14D joint states / already-logged 14D actions, published 50 Hz, Apache-2.0, exact publisher SHA `13ad96f5ed0e219e48c72471626a2e3d0eb5aeff`, sole Parquet SHA256 `aa61a929598903401e10d9b677bd2f2df480da0c912b6a85a27541028155fa1b`.
- Independent source-only audit PASS, [GitHub Actions #40279375687](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40279375687), before model fit. This audit scanned FIT data only; complete publisher Parquet contains all source episodes, but protected TEST values were not scored. FIT episodes0–24, DEV25–34, source TEST35–49 sealed.
- Original [12-model fit+first-score workflow #40280359434](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40280359434) **CI PASS**. **Twelve independent arm/seed model fits**: four arms (TAC-shared, TAC-isolated, velocity-GRU-shared, velocity-GRU-isolated) × three seeds (28001/2/3), **192 AdamW steps × batch8 × window32**, identical source FIT windows and loss values within paired starting weights. Original checkpoint/optimizer/RNG states for ALL twelve arms saved with SHA256 **BEFORE DEV rows were made visible to the scoring module**. [Original evidence artifact #12632067673](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40280359434/artifacts/12632067673), with all twelve original model states, original data/source/normalization manifest and original full per-episode first DEV scores. Full permanent source-linked model hashes and original numbers: [S1MOTION004-ORIGINAL-LEDGER.json](S1MOTION004-ORIGINAL-LEDGER.json). No re-fit or post-score hyperparameter tuning.
- Active trained/deployed parameters TAC=95,907, GRU=95,869 (0.040% gap). Both use **same actual causal constant velocity** based on genuine known x[t-1] and x[t], logged prior action u[t-1], same FIT standardization and .1 bounded correction. TAC native memory remains single 88+64+14=166 FP32 stream scalars=**664 bytes** excluding metadata/runtime/weights. Risk score is high *measured joint motion proxy* only; NOT genuine accident, event hazard, episode success, decision cost or policy authority.
- No fully matched training FLOPs claim! The supported Torch profiler subset (seed28001 FIT one batch) measured 277.79M for TAC-shared, 276.55M TAC-isolated, 138.03M GRU-shared and 136.76M GRU-isolated; FLOP reporting **excludes unsupported operations**, so numbers CANNOT be used as fully accounted training compute. Original per-seed CPU fit seconds and partial profiler metrics are stored in original score artifact. Paired within-TAC comparison is much better controlled than cross-architecture comparisons.

## Original real DEV — 2,550 measured next states per frozen model

All numbers source-coordinate 14D next-state MSE. The **same 10 genuine episode identities**, not 2,550 IID independent experiments. Constant velocity MSE **6.6024730648e-6**, persistence **9.0325978083e-5**, source high-motion prevalence **9.0588%**.

| Original seed | Velocity-only | TAC-shared | TAC-isolated | GRU-shared | GRU-isolated |
|:---|---:|---:|---:|---:|---:|
| 28001 | 6.602473e-6 | 6.969331e-6 | **6.513289e-6** | 6.599611e-6 | 6.456505e-6 |
| 28002 | 6.602473e-6 | 9.032717e-6 | **6.556679e-6** | 6.558484e-6 | 6.452301e-6 |
| 28003 | 6.602473e-6 | 7.065583e-6 | **6.527340e-6** | 6.579980e-6 | 6.431728e-6 |
| Mean (three seeds) | 6.602473e-6 | 7.689210e-6 | **6.532436e-6** | 6.579358e-6 | 6.446845e-6 |

**Point task positive:**
- TAC isolated vs same-seed TAC shared: relative MSE improvements **+6.54%, +27.41%, +7.62%**, all three; pooled seed-mean improvement **15.04%**. This is the DIRECT controlled intervention finding from real recorded outcomes; not an independent robot/site result.
- TAC isolated vs non-neural constant velocity: improvements **+1.35%, +0.69%, +1.14%**; original preregistered >=1% gate fails in seed28002. **It is a small but genuine source-DEV mean gain, not real-world qualification.**
- Within this one source, ten DEV episode-cluster descriptive bootstrap (20,000 resamples, same episode indices jointly across all 3 seeds) gives **13.32%-17.04%** range for the seed-averaged MSE gain vs TAC shared. *This interval is conditional on one source/task/split and the three frozen seeds; it is NOT cross-site, cross-task, independent-robot or population confidence evidence.*

### High joint movement = descriptive proxy, NOT real hazards

| Seed | TAC-shared AUPRC | TAC-isolated AUPRC | GRU-shared AUPRC | GRU-isolated AUPRC |
|---|---:|---:|---:|---:|
| 28001 | 0.33037 | **0.55971** | 0.69103 | 0.18134 |
| 28002 | 0.37472 | **0.56182** | 0.67429 | 0.22972 |
| 28003 | **0.55768** | 0.48683 | 0.68575 | 0.15566 |
| Mean | 0.42092 | 0.53612 | 0.68369 | 0.18891 |

- TAC-isolated loses AUPRC **12.70%** relative to TAC-shared in seed28003, violating the frozen **maximum 10% event-proxy AUPRC regression** guard; other two TAC seed AUPRC scores improve.
- GRU-isolated loses most proxy ranking signal in all three seeds despite better point MSE, proving that unconditional gradient separation trades away important supervised classification representation in this setup. Do not claim universal objective isolation benefit or actual adverse-risk competence.
- Original measured true high-motion MSE isolated TAC was **2.06846e-5 / 2.08431e-5 / 2.08502e-5** versus velocity **2.10982e-5** in all seeds, passing the pre-registered no >5% harm check. But G2 seed28002 and G3 seed28003 still FAIL. Study FULL GATE **FAIL (1 of 3 pass)**.

## Original residual-correction diagnostic

For each original frozen checkpoint on first opened DEV, let
- `b=2*x_t-x_(t-1)` causal observed velocity,
- `c=prediction-b` bounded learned correction,
- `r=actual_(t+1)-b` measured true residual.

Exact real `ΔMSE=E[c²]-2E[c·r]`. Pooled three seed × ten episode source means:

| Arm | Mean correction energy E[c²] | Signed alignment E[c·r] | Net MSE change vs velocity |
|---|---:|---:|---:|
| TAC-shared | 1.05597e-6 | −1.53828e-8 | **+1.08674e-6 harm** |
| TAC-isolated | 4.63783e-7 | **+2.66910e-7** | **−7.00364e-8 gain** |
| GRU-shared | 1.26864e-7 | +7.49892e-8 | −2.31143e-8 gain |
| GRU-isolated | 2.30916e-7 | +1.93272e-7 | −1.55628e-7 gain |

This decomposition explains the **observed error arithmetic**, not the cause of optimizer behavior. Gradient-path isolation is the deliberate intervention, but competing data/optimization and other structure still warrant independent replication. Detached features remove auxiliary gradients from the TAC encoder/memory but auxiliary heads continue to learn; same-weight before-fit point, logit and scale tensors were unit-verified as numerically equal. A full Lean/FP32/AD source refinement remains unproved.

## Scientific status and planned follow-up
- Source-only provenance, genuine observations, FIT-only statistics, same-seed identical initial weights, isolated gradient graph, chronological prior-action admission, stream/batch parity, original weights SHA before DEV, fixed first score: **PASS engineering/CI**.
- Original published **full predictive/robustness/proxy gate FAIL 2/3 seeds**, though point-only paired learning improved 3/3. **Do NOT promote System1** to achieved real-world general low-latency intelligent autonomy or production policy.
- The next research needs **partially decoupled auxiliary training**, not indiscriminately shared or indiscriminately detached: preserve classification signal without sacrificing source-world residual alignment. Test a *controlled* auxiliary gradient budget and its actual AdamW applied effect, jointly against unchanged original shared/isolated baselines, with equal active parameters, exact source/time train budgets and source-agnostic evaluation on an authenticated NEW real robot task. New source/splits preregistered before any scoring; keep towel DEV and towel TEST sealed for further tuning. Add independent genuine adverse/success decision costs, cold replay, full compute budgets and PNDS numerical refinement.


## Independent original 12-checkpoint forensic verification — complete

A wholly read-only [original cold-replay run #40281149614](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40281149614) **CI PASS** restored **ALL 12 SHA-pinned original model/optimizer/RNG artifacts** from the first training run, verified the exact publisher file checksum and exact FIT-only normalization, replayed already-opened original DEV, and separately inspected a genuine FIT batch's gradient pathways **without any optimizer update or re-fitting**. All twelve original high-motion AUPRC values reproduced exactly; maximum observed physical source-unit MSE drift was **5.0673e-14** across original models. This is not a proof of bitwise deterministic operation on every CPU, nor an independent new real-data test. [Read-only audit artifact #12631069872](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40281149614/artifacts/12631069872) preserves per-model source-gradient and replay drift receipts. Original scientific gate unchanged.

**Actual frozen FIT gradient diagnosis**, same first source window per seed, for the **TAC associative memory-gate parameter**:
| Original seed | Shared point gradient norm | Shared auxiliary gradient norm | Auxiliary/point norm | Isolated auxiliary reach | Isolated point reach |
|:--|--:|--:|--:|:--|:--|
| 28001 | 6.46e-8 | 1.78e-5 | ~276× | **none** | nonzero |
| 28002 | 2.12e-7 | 2.47e-4 | ~1,168× | **none** | nonzero |
| 28003 | 1.55e-8 | 9.15e-5 | ~5,886× | **none** | nonzero |

These are **weighted auxiliary objective gradient magnitudes on a single frozen real FIT batch**, not a proof that AdamW's applied preconditioned update is antagonistic, and not representative of every time/window. The gradient separation is independently verified in the live source through actual `torch.autograd.grad`: head gradients continue, auxiliary-to-memory gradients are absent, point-to-memory gradients continue. This turns the prior abstract suspicion of a competing auxiliary gradient pathway into a verified *local architectural mechanism* with observed benefits and classification tradeoffs, while the strict original full success gate stays failed.
