# MODEL01-TAC-CAL-COVERAGE-BUFFER-001 — original source results

**Evidence tier:** post-hoc **EXPOSED SINGLE-HOUSEHOLD DEV**, real original UCI374 appliance electricity data. **Training:** none in this study. Original 103,252-parameter fast/slow/associative neural checkpoints from #502 completely frozen; original last 20% protected UCI rows never parsed. Source available-at and generalization still unknown.

## Execution / trace

- [Original real-source CPU Actions run #40497361468](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40497361468): succeeded with preflight tests and original source read/scoring. The exact source JSON is Git-owned at `research/source-evidence/MODEL01-TAC-CAL-COVERAGE-BUFFER-001-ORIGINAL.json` (Git blob SHA256-of-blob not the same as original source: blob `3499a0f2be8399071bb32f1c5d222a2306ea4114`). Original publisher ZIP SHA256 is `2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a`.
- Frozen checkpoint hashes, full optimizer states/RNG/sample-order in the parent original native three-seed Git bundle; [clean native replay #40488762280](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40488762280) passed.
- Three registered *reported but not selected* source-phase signed-quantile intervals nominal mass 90%, 94%, 96%, all fitted from the first chronological 60% CAL. Two subsequent distinct CAL subperiods of 20% each and the 1,968-case already exposed DEV are reported without choosing a winner based on DEV. Quantiles are empirical, temporally dependent and do not have classical i.i.d. marginal coverage guarantees.

## Mean original DEV metrics (averages across three trained fixed seeds)

| Scheme | 90% evaluation empirical coverage | Interval score lower better | Mean interval width Wh | All seeds >=90% on both CAL final audit + exposed DEV? |
|---|---:|---:|---:|---|
| Original full-CAL symmetric 90% | 90.72% | 436.83 | 152.55 | Not evaluated on same CAL subset |
| Four-phase signed 90% | **89.21%** | 338.37 | 215.89 | **NO** |
| Four-phase signed 94% | **93.63%** | **370.65** | 300.47 | YES on reused history only |
| Four-phase signed 96% | **94.87%** | 404.02 | 358.32 | YES on reused history only |

For the phase-94 option, mean DEV interval score is **15.15% lower** than original symmetric, mean interval width **96.96% wider**, and empirical coverage **2.91 percentage points higher**. Wider intervals are a **real operational cost** and may be inappropriate for narrow-alert workloads. The original point predictions, MAE and RMSE are identical because no neural parameter or forecast center changes.

## All-seed 94%-buffer source outcomes

| Original neural seed | Last CAL 20% coverage | Exposed DEV coverage | Exposed DEV interval score | Exposed DEV mean width |
|---|---:|---:|---:|---:|
| 1701 | 93.03% | 93.90% | 367.96 | 304.10 Wh |
| 1702 | 94.04% | 93.60% | 372.59 | 299.74 Wh |
| 1703 | 93.28% | 93.39% | 371.41 | 297.58 Wh |

The prior phase-90 candidate failed nominal90 coverage on CAL final audit and DEV for all three seeds; the present **buffer-94 passes that descriptive historical criterion on all three**. This **does not reverse the previous study's fail verdict** or prove a new independent success. It establishes that widening signed-tail offsets helps this already-repeatedly-analyzed same-household source, at substantial width cost. A phase-96 option is yet wider with worse interval score (albeit more empirical coverage).

## Hard research/operational distinction

**Research observation:** historical source-derived coverage and score tradeoff characterized, and reproducible. **Independent scientific qualification:** FAIL / NOT ESTABLISHED (single reused source, known DEV results before this buffer hypothesis, source available-at unverified, overlapping outcomes, no external calibration). **Deployment readiness:** NO. Do not present 94% buffer as a calibrated 90%-guaranteed predictive API, autonomous action/physical controller, or proven new-household skill. No TAC long-memory or native associative advantage was isolated; no new training and no matched ~100k architecture comparison.

## Next decisive evidence

Freeze a target-matched **new** real household or customer site with independent documented sensor observation/provenance and suitable legal rights, explicit source `available_at` clocks, cross-household PIT splits and per-site loading/scaling contract. Compare point MAE/RMSE and full forecast uncertainty coverage/interval score (including width and temporal gaps) against strong per-site persistence and classical models on previously untouched cases. Then, if scientific gates pass, investigate actual customer adoption and operator cost.
