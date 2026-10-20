# ME-011 — Authentically observed cross-domain quantile forecasting

**Preregistration:** 2026-10-20, before new ETTh1 training or any ETTh2 inspection. Existing ME-004–ME-010 positive and negative evidence preserved without replacing TAC-OSM architecture, neural CASM, CDL-PHR, BDH, role–filler memory, or PNDS Lean formalization.

## Independent physical data
Official real Electricity Transformer Temperature dataset `zhouhaoyi/ETDataset` (<https://github.com/zhouhaoyi/ETDataset>), collected from transformer instrumentation in China, distinct from the one MetroPT air compressor. Hourly ETTh1 and ETTh2 from two stations; physical oil temperature (OT) and six measured load values (HUFL,HULL,MUFL,MULL,LUFL,LULL). Source: `https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv`. Pin existing documented official source MD5 `8381763947c85f4be6ac456c508460d6` and record source SHA256. Abort on hash mismatch, timestamp irregularity, unexpected schema, missing values or bad row count. Source published, public benchmark, not a new data collection by this research. Do not interpret observational prediction as repair efficacy.

## Source-exposure registry
- **TRAIN, ETTh1:** row indices [0,8760) only; first ~12 months. TRAIN-only feature mean/std; OT baseline in physical degrees C.
- **Development/calibration, ETTh1:** [8760,11640). Validation can be inspected for engineering diagnostics; no model/seed selection, loss search or hyperparameters picked from these scores.
- **Prospective frozen ETTh1 TEST:** [11640,14520), 4-month chronological later period **NOT read by original training code**.
- **Prospective cross-site ETTh2 TEST:** [11640,14520), other physical transformer station; **NOT downloaded or opened by the original training code**; apply frozen ETTh1 normalization and weights with zero fine-tuning.
- Rows 14520 onward remain reserved by both test and training code. For prospective test, verify original weights/hashes before either ETTh1 test period or ETTh2 source is accessed. No repeated exposure with tuned weights.

## Native engineering question
Can TAC-OSM's stable identity-first two-timescale recurrence produce **better-calibrated probabilistic predictions** on an *independent authentic operational domain* compared with an equally trained GRU? Do not change TAC recurrence to SSM. Input 96 past hourly observations × 7 genuine features + missingness masks; two horizons 24 and 168 hours ahead, target OT signed delta relative to most recent observation, 3 monotonic predictive quantiles (10th, median, 90th), trained with pinball loss. No future covariates or targets enter model input.

Native state: `z_t=(1-g_t)z_{t-1}+g_t tanh(Uu_t+Vz_{t-1})`; slow state `s_t=(1-g_t^s)s_{t-1}+g_t^s tanh(W[s_{t-1},z_t])`, hidden width 32. Exactly same **mathematical recurrence family** shown to work in ME004/ME008; do not reuse trained MetroPT weights (input/physical task differs). GRU control hidden width 40. Both use noncrossing quantile head `q0.1=mu-softplus(a)`, `q0.5=mu`, `q0.9=mu+softplus(b)`. Noncrossing quantile inequalities are structural; coverage is **NOT guaranteed by them**.

Same three seeds 11001,11002,11003. Fixed 450 AdamW steps, batch64, lr .001, decay 1e-4, clipping 1; original observations, train-only normalization. Source size and runtime recorded. Network parameters may differ; report exact counts and measured CPU time, never claim FLOP equality. Archived checkpoints must be hash-verified before later test.

## Preregistered TEST metrics and gate
Primary **median forecast MAE in degrees C** aggregated over two horizons. Report separate 24h/168h MAE, mean three-quantile pinball, empirical central 80% prediction interval coverage (target 0.80), per-horizon coverage, and mean interval width in degrees C. Compare persistence (constant last OT) and 24h rolling-average benchmark, both using only information available at forecast time. Quantile calibration must be compared with the target empirically; negative sharpness/calibration evidence retained. Evaluate on both ETTh1 future test and cross-site ETTh2 separately, not cherry-picked combined winner. A TAC win against GRU and persistence **on both sites across all three paired seeds**, with materially improved pinball and close-to-target coverage (difference <=0.10) is a first screen for researching the novel recurrence. Failure is not reinterpreted as architectural success.

## Scientific limits
Independent domain and a second site strengthen generalization tests, but only two stations, two horizons, no physically grounded control actions or repair-outcome causal identification; no commercial/state-of-the-art conclusion. A strong GRU win means revisit optimization and update-interference rather than invent more mathematics. Longer-term master plan remains intact: entity discovery, self-organizing sparse connectivity, verified executable memory, neural CASM, CDL-PHR plastic relevance, oscillator gates, continual plasticity, adaptive reasoning, Veritas safety and PNDS Lean trust evidence are tracked but only promoted with appropriate real-world data.

No auto-merge; preserve all checkpoints, negative results and source provenance on GitHub.
