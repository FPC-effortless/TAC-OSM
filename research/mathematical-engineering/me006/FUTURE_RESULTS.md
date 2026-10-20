# ME-006F — Original frozen out-of-time real forecast results

**Outcome:** The direct future-trained readout, not bounded Hebbian association, is the best of these ME-006 candidates. No architecture promotion, no merge, no commercial claim.

- Preregistered period: **2020-03-09 00:00:00 inclusive to 2020-03-16 00:00:00 exclusive**, following the source windows explored by ME004 and ME003.
- Original successful [GitHub Actions run 39380961862](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39380961862), original first-exposure commit `3fe4a1771ce9930faeba01813207a6afbd4e8b87`.
- Original artifact `12244554520` containing raw `future-results.json` and logs.
- Exact original ME004 weights from run `39357593422`, exact ME006 adapter weights from run `39380424029`, SHA-256 checked against their respective original ledgers. **ZERO** adapter refits/optimizer updates or future-period hyperparameter changes.
- Genuine MetroPT compressor observations: **48,442** chronological sensor rows within the registered week; **3,897** valid future prediction windows with matched real readings, original TRAIN-only normalizations, 12 authentic physical sensor channels, horizons 12 and 60 observation steps. Source transcript cutoff was honored; no source rows at/after 2020-03-16 read.

## Measured macro error — ORIGINAL frozen models (lower is better)

| Model | Seed 240401→606001 | Seed 240402→606002 | Two-seed average |
|---|---:|---:|---:|
| Persistence (same for both) | 0.748368891 | 0.748368891 | 0.748368891 |
| Original pretrained native TAC | 0.683453170 | 0.688586558 | **0.686019864** |
| Direct future-trained adapter | **0.672430903** | **0.677994492** | **0.675212698** |
| Bounded Hebbian associative adapter | 0.682788519 | 0.686747254 | 0.684767886 |

**Result:** Direct adapter improves over original TAC by **1.58%** average MAE and beats it in **both** frozen original seeds. Associative improves slightly over TAC baseline in both seeds (mean 0.684768 vs 0.686020), but is **1.40% worse than direct**, and the original training used approximately 2.44× more CPU fitting time than direct.

**Changed-value analysis:** The future evaluation has **10,606** significant target changes (**5,616** positive, **4,990** negative). Changed-value magnitude MAE:
- Original native TAC: **1.98715** / **1.99674**.
- Direct: **1.92263** / **1.93021**.
- Hebbian associative: **1.98381** / **1.99224**.
Direction accuracy:
- Direct **82.15%** / **82.26%**;
- Associative **81.51%** / **83.32%**.
These direction metrics are conditional on **observed true changes**, not a deployable advance detector; there is no externally independently verified action-outcome evidence.

## Scientific decision

- The **direct future-supervised adapter** now has limited one-source out-of-time corroboration beyond the exploratory data on the same compressor. Keep as a research forecasting component.
- **Reject the current bounded Hebbian associative route for practical promotion.** It does not beat the future-trained direct alternative, despite being mathematically bounded. Do not quietly adjust its learning rate based on this opened future week.
- No superiority claim against a capacity/compute-matched GRU/LightGBM, no causal CDL interpretation, no autonomous entity discovery, no learned executable composition, and no System 2/decision-cost claim. The real test is one equipment stream and two seeds.
- Keep every idea and negative result in the unified master plan. PNDS Lean theorem build covers explicit admission and convex/refinement contracts, **not** learned predictive accuracy or real receipt authenticity.

**Fastest next work:** Use the direct forecasting improvement as a validated research candidate and test against a capacity/compute matched GRU on another independent domain/source. Improve signal-specific change direction/calibration and feed the *qualified* predictive state into the already trained DEV067 action/STOP agent only with real logged actions, verified outcomes and cost evidence.
