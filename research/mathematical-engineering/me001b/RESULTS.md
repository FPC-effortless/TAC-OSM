# ME-001B — Original real SCANIA results (FAILED native-architecture gate)

**Original training:** GitHub Actions run [39334730465](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39334730465), original source commit `13fc199642845698fa4d2d8dd623079731c3a47c`; evidence artifact `12225908024` (me001b-original-scania-train-internal).

**Scientific qualification:** Real authentic sensor observations, publisher TRAIN only, internal vehicle-heldout split. **NO native TAC promotion.** No new model data, trained model fitting or score tuning introduced by this analysis. Historic SEQ010 and publisher VALIDATION/TEST remain untouched.

Source: SCANIA v3 official TRAIN member SHA256 `e01cb0bd87dfab4c9dbad215d51d81282fc0d413be96d6f819ea872fb7a3c715`, original publisher ZIP SHA256 `6fdf63d15805f4ff429e6cd3172c4bd1634ca5c6d293406a79553c56c07db483`. The TRAIN member identity, not ZIP container identity, is the reproducibility criterion. 1,122,452 real operational readouts; 23,550 original TRAIN trucks; new deterministic internal partition: 18,848 train and 4,702 heldout trucks, 47,097 sampled train windows, 4,702 validation windows. First 24 observed sensors, at most eight prior readouts, missingness masks; same inputs/targets and signed-log units for all arms. Original checkpoint artifact contains nine models, and all **nine checkpoint SHA256 hashes matched** the corresponding ledger entries in a separate artifact inspection.

## Primary macro signed-log MAE (lower better)

| Registered arm | Seed 119001 | 119002 | 119003 | Three-seed mean |
|---|---:|---:|---:|---:|
| Persistence (untrained) | 0.041950082 | 0.041950082 | 0.041950082 | **0.041950082** |
| Linear eight-step history, 9,240 params | 0.052326411 | 0.043252328 | 0.062681191 | 0.052753310 |
| Global GRU, 8,664 params | **0.039288367** | 0.043197758 | **0.041357847** | **0.041281324** |
| Native TAC four-assembly, 4,397 params | 0.042788542 | 0.043954008 | **0.041901476** | 0.042881342 |

Original study FAILS every-seed-better-than-persistence gate for TAC (1/3 seeds beat persistence); GRU also fails that gate (2/3 seeds). Relative to persistence, mean native TAC is **2.22% worse**, GRU is **1.59% better**, linear history **25.75% worse**. No statistical independent qualification or superiority established; unequal parameter counts and approximately 0.65 vs 4.16 seconds mean training wall time for GRU and native TAC respectively prohibit a fair architecture-efficiency claim.

## Important change-selection findings
Training-only 90th percentile feature-specific absolute-change thresholds (floor 1e-4) selected 6,379 heldout paired sensor changes over 24 features. Means over registered seeds:

| Model | Change macro MAE | Change direction accuracy |
|---|---:|---:|
| Persistence (zero-residual) | 0.479896933 | defined as 0.0 in current script |
| Linear history | 0.413772867 | 0.915190469 |
| Global GRU | 0.430563570 | 0.987458849 |
| Native TAC | 0.439353995 | 0.829858390 |

**Metric warning:** The original zero-residual persistence baseline gets direction=0.0 by an implementation convention rather than a comparable probabilistic or signed-direction classifier. The very high direction accuracies, including after resetting native state (1.0 with much worse overall MAE), warrant class-balance and naive sign-prior diagnostics. Do not interpret them as demonstrated direction-discrimination skill. Change MAE is based on hindsight-identified target changes, not deployable foreknowledge of when a change will happen.

**State reset control:** After zeroing the native state halfway through the eight readout window, macro MAE rises to 0.12856–0.13525 versus 0.04190–0.04395. Therefore carried state matters for this model's output, but this alone cannot establish it stores the correct identity/history.

## Decision and next experiments
- Status: **REAL exploratory native TAC FAIL / no architecture promotion**.
- Historical frozen artifact and all negative seeds retained; no refitting of the same holdout for a favorable result.
- Independent source+checkpoint **recorded-only replay** is the next verification, not retraining.
- Design a new preregistered study using fresh real vehicle/future-time cohorts, compute-matched and tuned baselines (persistence, regularized drift/linear/LightGBM, GRU, native TAC), plus genuine rare-change sign class balance and no-leakage checks. Test state-only interventions and learned-routing attribution.
- Entity identity discovery, CDL-PHR, explicit executable composition, real action outcomes, causal counterfactuals and commercial savings are NOT tested by ME001B.

**No merge. No promoted checkpoint.**
