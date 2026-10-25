# EWA001 — Genuine real-source evaluation of online outcome-weighted memory

**Executed 2026-10-25.** **STATUS: TRAINED_ME097_FROZEN / ONLINE_NONPARAMETRIC_UTILITY / EXPOSED_DEV_ONLY / REGISTERED_FULL_GATE_FAIL.** Original [preregistered protocol](PROTOCOL.md) committed at84d2684 before the first attempted run. Frozen real-trained model family is ME097, 95,874 learned parameters, source FIT_INDEX SHA256 verification. No neural training, checkpoint change, old qualification rewrite, CED assumption, source-TEST access, or new hardware action.

**Authoritative first complete scored execution:** [GitHub Actions #40209655856](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40209655856), SUCCESS; original JSON artifact **12605049012** named ewa001-original-five-checkpoint-exposed-real-result. Five exact ME097 model checkpoints SHA256-authenticated, genuine [original ALOHA source](https://huggingface.co/datasets/lerobot/aloha_mobile_cabinet) SHA256 a3409d3282b2cf9a94d32b301756fbd15dca5862cc1c32973c7316dd8853c57a, 15,104 FIT observations used for causal cache experience per seed, **13 already-exposed DEV episodes / 3,328 original one-step future observations per seed**. The neural model previously trained on FIT. The TEST numeric outcomes were not materialized/scored. Original FROZEN weights stayed unchanged.

## Fixed-preregistered comparison: five original checkpoint seeds

Standardized 14D next-observation MSE; lower is better; all methods use the same source chronology. Utility/neutral share identical learned source keys, source FIT/DEV, 256 cache slots, candidate matching cap=8, residual correction alpha=.25 and 8-plane radius-one LSH. Utility differs in after-outcome loss-based EMA, ranking and eviction. Persistence/velocity are standalone no-fit comparisons on the same ground truth.

| ME097 checkpoint | Frozen original MSE | Similarity-only cache | Outcome-utility cache | Real constant velocity | Real persistence | EWA registered gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
|97001|0.000514045|**0.000338191**|0.000367724|**0.000031119**|0.000616392|FAIL|
|97002|0.000526525|**0.000348816**|0.000393633|**0.000031119**|0.000616392|FAIL|
|97003|0.000501606|**0.000333750**|0.000348829|**0.000031119**|0.000616392|FAIL|
|97004|0.000533509|**0.000352860**|0.000413415|**0.000031119**|0.000616392|FAIL|
|97005|0.000485172|**0.000324672**|0.000335706|**0.000031119**|0.000616392|FAIL|
|**Five-checkpoint arithmetic mean**|0.000512171|**0.000339658**|0.000371861|**0.000031119**|0.000616392|**FAIL 0/5**|

On the **previously exposed** real DEV data, neutral similarity residual retrieval improves checkpoint-only MSE **33.68%**. This does **NOT** mean trained native associative-memory impact was isolated: the original checkpoint remains the same and the external residual cache receives real FIT next-state labels. Conversely the outcome-utility arm regresses **9.48%** vs similarity-only and is worse in all five seeds (+8.73%, +12.85%, +4.52%, +17.16%, +3.40% error). It is still 27.40% below original checkpoint-only MSE, but that is not sufficient to satisfy the specified incremental utility hypothesis. Even the better similarity cache is **10.9× the MSE of constant velocity**, so neither qualifies a strong physical model or controller.

Recorded DEV movement-rich samples: **393 of 3,328 / seed**. Neutral/utility both improve over frozen on this subset, yet utility regresses against neutral in all five original seeds. The registered usefulness gate requires improvement over BOTH controls, and registered LSH scoring <32 full candidates/query; neither holds. No source-independent physics/actor-site result or confidence claim from repeated model seeds.

## Query/index efficiency and utility outcomes

| Frozen checkpoint | Mean cosine candidates/query, neutral | Mean candidates/query, utility | Neutral exact top8 selected overlap | Utility selected overlap |
| --- | ---: | ---: | ---: | ---: |
|97001|93.03|75.40|99.00%|25.52%|
|97002|143.91|135.04|98.99%|21.08%|
|97003|99.74|83.74|97.49%|29.25%|
|97004|99.51|85.14|98.64%|21.57%|
|97005|77.17|57.21|97.09%|33.71%|
|**Mean**|**102.67**|**87.31**|**98.24%**|**26.22%**|

LSH probes exactly **nine** signatures per query (original 8-bit plus eight one-bit neighbors). The bucket collision size is still much larger than the <=32 registered cost target. **Important metric meaning:** the original comparison reports overlap between the FINAL selected top8 and exact top8 cosine-nearest. In the utility arm, utility-based re-ranking also changes the selected top8; thus its low overlap is **not an isolated measure of hash-index recall** and must not be described as 74% index misses. This is a limitation of the registered metric, with any separate hash-only recall study subject to new source/predeclared analysis. No post-score filtering of cases, hand-tuning LSH planes, or parameter changes authorized.

Both memory arms consume an external cache of **256×(8 key + 14 normalized residual + 1 utility)×4 = 23,552 additional float bytes**, plus slot provenance, timestamps, hash buckets, Python allocations and other runtime overhead. Original native primary 608B continuous recurrent state and all ME097 neural weights stay unchanged, but the **complete system does not have a 608B state footprint**. Observed lookup CPU accumulates approx 1.6–2.3 seconds over FIT+DEV events/seed; utility memory write/feedback overhead around 2.7 seconds vs around 0.6 seconds neutral. These are non-independent runner process-time observations, not a real-time embedded SLA and do not include full deployment memory/network costs.

## Causal learning interpretation

At each real recorded episode frame t: original model observes current joint vector x_t and recorded lagged action a_(t-1), computes its causal 8D representation and original next-state prediction. Each 256-slot cache uses only previously verified (key, model residual) examples. Only **after** both cache forecasts have been issued does it receive recorded x_(t+1) and update any previously retrieved memory's utility. Corrected forecast uses the predeclared bounded weighted residual. The memory update is *nonparametric online learning* from observed verified outcomes. There is **no retraining of the 95,874 ME097 neural parameters** and no eligible causal do(a) alternative-action estimates. Utility is a per-record single-memory incremental loss proxy: its positive/negative values may reflect misleading near-zero denominators, co-occurrence, utility concentration and interactions, not causal business/physical memory value.

This demonstrates it is **possible to build** delayed usefulness feedback and fast indexed event matching, but this first design does **not** make the model more useful than neutral source-similarity memory. Practical real downstream performance still strongly favors a velocity prior.

## Execution integrity and retained failed pre-score attempts

- First new workflow [#40209426834](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40209426834) FAILED before real-source experiment: unit test incorrectly attempted to read stats while pending unverified observation; model code correctly failed closed. The test expectation was corrected and original failed CI retained.
- The next [#40209527278](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40209527278) passed delayed feedback unit checks but FAILED while assembling a progress-metric count (missing raw hits field), before a completed scored result was emitted. An accounting field was exposed; original core math, immutable checkpoint and experiment thresholds unchanged. No original score JSON was published from this failure. It did execute source reads, but did not result in a complete scored five-seed study; this is disclosed, not called a clean single attempt.
- First successful complete trial is #40209655856, original measured results above. Any repeat must be exact original replay/audit only, not changed coefficients or new-seed trial disguised as continuation. The CI archive preserves the original complete per-seed/per-episode source scores.

**Verdict:** registered **FAIL** (0/5). No real-world qualification, native memory-use attribution, general intelligence, new trained model family, device deployment or independent validation. Original six active model families and history remain preserved.

## Next research constraint, based on this result

Avoid blindly increasing utility weight or modifying the original scored version. A separate preregistered, fair real-data study should isolate: (i) candidate-matching quality before utility re-ranking, (ii) measured per-memory damage from matched real outcome under bounded contributions, (iii) nonstationary forgetting/age and false-success reward credit, and (iv) strong physically meaningful velocity/servo residual baselines at same factual information. Use a fresh authorized real source/time test and one complete model fit where actual trainable architectures are compared fairly under 95k–105k active parameter, <=1% paired disparity and equal source/data/compute/tuning. The next operator must preserve native scan causality or account explicitly for sequential read-dependent updates. No full-new-Lean or calibrated belief result is claimed here.
