# SYS1-SYS2-REAL-002 — real logged action/outcome gate

**Registered:** 2026-10-19. **Status:** PROTOCOL/SOURCE GATE, not qualified System 2 performance.

## Research question
Can learned TAC-OSM state + action-consequence planning beat equal-information, compute-matched reactive controllers on authentic, sequential, logged recommendations, where actions are displayed videos, observations are preceding user behavior, and outcomes are subsequent recorded clicks/watch duration?

## Authentic data
Publisher: Gao et al., *KuaiRand* (CIKM 2022), Zenodo original version https://zenodo.org/records/10439422, https://kuairand.com/, original dataset license respected. Publisher KuaiRand-Pure tar file MD5 **0820331067a3784d9691136f772b35a7**; Zenodo source must be SHA256 logged also. KuaiRand-Pure contains randomized item exposures with actual engagement; it **does not contain complete continuous per-user interaction histories**. It is acceptable for publisher provenance and a *one-step factual observed-reward baseline*, NOT for complete multi-step planning. The full original KuaiRand-1K (publisher MD5 **6b0b9c8222d67fcd4c676218edca3f1f**, approx. 1.1 GB compressed) is required before multi-step qualification: original standard before/after plus randomized logs must be linked by user ID and time, with gaps/competing events audited.

## Strict study stages
**Stage 0: source qualification and descriptive baselines.** Download official Pure archive, check exact MD5 and record SHA256, extract allow-listed train/standard/random CSV files only, validate schema and real exposure/outcome/time columns. Training on standard-history 2022-04-08..21; validation on *randomized* exposures dated 2022-04-22..27; future-period data 2022-04-28..May-08 may be schema-audited but no future labels are used to train or score. Report observed click-rate, global, train-only user and train-only item smoothed estimates (Brier/logloss), support counts, and distribution shift. No raw user logs in CI artifacts.

**Stage 1: full timeline.** Acquire publisher KuaiRand-1K, source-hash all allowed files, order per-user observed exposures by timestamp; verify identifier mapping, same-day ties, gaps, filtering and action candidates, 15 logging policies, missing propensities and which random actions truly have known probabilities. Define train-only, chronological validation, and sealed time/user TEST. No future exposure, reward, item popularity, or target leakage.

**Stage 2: factual prediction / System-1 proxies.** Train action-conditional 1-step rewards and state-transition models. Evaluate Brier, logloss, GAUC, calibration, outcome prediction by user and chronological window; compare persistence, popularity, logistic/GBDT, GRU, routed GRU, n-gram, TAC pretrained+fresh adapter, TAC scratch and no-CASM, accounting equal data and inference/training compute. Native DEV067 maintenance input schemas cannot be treated as general recommendation policies without learning a declared adapter; no zero-shot superiority claims.

**Stage 3: System-2 necessary conditions.** Candidate actions must have meaningful multi-step dependency and testable *oracle-over-greedy headroom* from observed data; compare no-plan reactive vs learned rollout horizon 2/3+ vs context-matched GRU planner, receding horizon variants and oracle using only recorded known outcomes. Run identical-action and head-reset ablations. Off-policy policy-value estimation only with defensible logging probabilities and overlap; report IPS/SNIPS or doubly robust bounds, effective sample size, support thresholds, sensitivity to exposure selection and unobserved confounding. Evaluate only observed factual outcomes when these assumptions fail. Do not treat a next-click predictor or value-head dependence as evidence of actual improved planning.

## Preset guards
- No synthetic world/model accuracy counts as real-world qualification; fixtures allowed for ingestion/unit tests only.
- Never extrapolate the next real-world state from an off-policy action without identified support.
- No promotion if a strong simple baseline matches or wins, if policy counterfactuals are unidentified, or if held-out multistep gains do not pass uncertainty/reliability gates.
- Each checkpoint/source manifest retains exact SHA256, original GitHub Actions run, dataset time range, seed, frozen/retrained status and provenance. Always preserve negative results.
- Stage 0 deliberately does **not** evaluate pretrained TAC-OSM weights or claim System 1/System 2 capabilities.
- Source dataset user-level identifiers remain internal; never release raw logs in CI result artifacts.
- No interactions with a live recommendation service or users; this is strictly offline logged data.

## Links and previous evidence
- Earlier frozen-checkpoint System 1/2 diagnostics: PR #241. Native DEV067 policy is an action scorer with a future-output head; no explicit multi-step search.
- Real-world-only promotion policy: PR #237.
- Previous authentic Scania baseline failure: PR #239 (affine AR worse than persistence); not an action-outcome policy benchmark.
- KuaiRand publisher: https://github.com/chongminggao/KuaiRand and DOI https://doi.org/10.1145/3511808.3557624

**Acceptance status:** Stage 0 not a planning test. Stage 2/3 unexecuted pending complete real-history acquisition, causal support and properly trained adapters.