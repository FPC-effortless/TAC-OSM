# ME-015G — Strongest GRU control on additional authentic never-trained user population

**Decision:** Engineered native TAC has the BEST **three-seed mean** Brier and log loss on a second genuine never-trained cohort, beating both strengthened/anchored GRU and the original unanchored GRU in three-seed average, but **FAILS** preregistered all-three-seed superiority against original unanchored GRU (GRU wins seed 14002). No scientifically robust universal GRU-superiority or commercial decision claim.

## Exact first-exposure provenance
- Successful original [GitHub Actions run 39400617213](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39400617213), first-exposure SHA `33e9082367f3e32c94d38d804a040ce9f952e4af`, original archive artifact ID `12254619086` (`me015g-first-original-strong-gru-disjoint-users`), original results JSON SHA256 `2c54a9da45d1ee782f180f6a0e330ba945c522aa2c16ff01d0d07fb73a9ee1bf`. Freeze CI against re-exposure after original result.
- Original fitted native TAC and anchored GRU archive [run 39398038257](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39398038257), artifact `12253856564` (15 real trained checkpoints). Original stronger unanchored GRU weights from [run 39394500641](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39394500641), artifact `12251666454` (6 original exact-size checkpoints). **All 21 original active-weight file digests verified before any real `uid%16==2` reaction was parsed**, and every model exactly **135,050 trainable parameters**.
- Source authentic KuaiRec.big_matrix from public original ZIP, original publisher MD5 `261550d472c48eff4990fb13c0e5bcf7`, SHA256 `c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c`. Only original users id%16==0 were in training; first heldout population id%16==1 was already opened separately in ME015F. This second test registered **disjoint never-trained users id%16==2**, 436 real users, **40,396** timestamp-valid Aug14–Sep06 2020 actual video-exposure/full-watch outcome windows. The full previous 32 actual user interactions may include previously observed earlier test-period reactions, not current/future results. Zero test model updates; no after-Sept06 source rows parsed. This is factual observational next-action response prediction, NOT a randomized action-counterfactual, independently verified benefit or calibrated policy uplift.
- Base models exact equal parameters, FIT examples, original optimizer updates, seeds, current action/video embedding and target. Original training CPU budgets **not identical**: native combined ~6.51 sec per fit, anchored GRU ~6.11sec per fit in ME015; original GRU ~7.14 sec in earlier separate original environment/run. Exact FLOP equality unknown; do not call these perfect equal-compute results.

## Original real future user2 outcomes (Brier score, lower better)

| Model | Seed 14001 | Seed 14002 | Seed 14003 | Mean |
|---|---:|---:|---:|---:|
| Native original TAC | 0.19478306 | 0.19329154 | 0.19765733 | 0.19524398 |
| **Native chronology+causal prior** | **0.19205585** | 0.19255521 | **0.19241744** | **0.19234283** |
| Anchored GRU | 0.19371340 | 0.19316617 | 0.19316472 | 0.19334810 |
| **Original strongest unanchored GRU** | 0.19238867 | **0.19190594** | 0.19656946 | 0.19362136 |
| Causally available prior32 real user full-watch rate | — | — | — | 0.19407570 |
| Original FIT video rate | — | — | — | 0.21781850 |
| Original FIT global watch rate | — | — | — | 0.21899500 |

Bernoulli log loss (3-seed mean, lower better): engineered TAC **0.56725968**, GRU anchored **0.5706955**, original GRU unanchored **0.57279914**, native TAC original **0.57644623**. Thus engineered TAC has the lowest mean Brier and logloss across the candidate models.

**Paired outcome:** Combined TAC beats anchored GRU on **3/3** original fixed seeds and original unanchored GRU on **2/3**. The original unanchored GRU beats TAC on seed 14002 by 0.00064927 Brier. TAC's average advantage over the original GRU is 0.00127853 Brier (~0.66% relative). Three seeds are too few for strong uncertainty or significance claims and the all-seed gate was **NOT met**, so do not promote “TAC beats GRU universally” or state-of-the-art.

## Engineering interpretation and next action
- The success is architectural **modification of the native trained predictor**, NOT adding learned experts/modules: correct fast/slow gate timescales plus factually available causal user base-rate log-odds. The same effects improved the old DEV, independent heldout user remainder1 and second independent remainder2 populations, with original source and test checksums.
- The paired seed2 defeat and ~0.6% training wall gap suggest optimization robustness remains; engineer recurrent gradient credit/learning-rate schedule and equal compute first, then use a **single new prospectively reserved real source** with strong unanchored/anchored GRU controls. Do not retune by selecting favorable already opened user cohorts.
- This test does NOT demonstrate semantic self-discovered identities, autonomous CASM planning, long-delay H512 memory, causal value of unshown actions, or independent PNDS/Lean proof of neural predictor accuracy. Keep all retained master-plan ideas and original negative experimental evidence.

**No model merge or production promotion. TAC remains a native predictive nonlinear recurrent world-state model, not an SSM.**
