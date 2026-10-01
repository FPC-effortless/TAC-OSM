# C5 noisy full-phase corrigendum

The measured values in `TACOSM-C5-NOISY-FULL-PHASE-001` remain unchanged. This note corrects how they should be interpreted.

## 1. Rank growth and admission loss are different quantities

At M=1024, dense-CDL K90 calibration is 17, and held-out dense coverage at K90 is 97.5% pooled (312/320 trials). The same held-out stream produces 65.31% sparse admission recall.

Therefore the current sparse funnel loses substantially more recall at the LSH admission stage than is lost between the dense CDL K90 shortlist and the target. CDL's observed cost is primarily expressed as increasing target rank and K90:

- P90 rank: 2 → 17 from M=64 → 1024;
- K90: 2 → 17;
- sparse rerank fraction: 7.62% → 2.72%.

The correct statement is that CDL has degraded ranking fidelity, while LSH is the larger observed recall loss at the chosen operating rule.

## 2. The 97.5% coverage is an empirical sample result

The result uses 64 held-out trials per seed per M. Across the 25 seed/M cells, 15 cells have 64/64 coverage and 10 have 60/64 coverage, producing the pooled 312/320 = 97.5% figure.

This is an empirical measurement under the registered sampling scheme. It should not be treated as a distribution-free guarantee.

## 3. gamma=0.776 is a full-range fit, not a statement about asymptotic behavior

The displayed P90 sequence `2,3,5,9,17` gives successive doubling slopes of approximately:

- 64→128: 0.585
- 128→256: 0.737
- 256→512: 0.848
- 512→1024: 0.918

The next audit therefore extends M to 2048, 4096, and 8192 and reports these local slopes before interpreting the fitted exponent.

## 4. The binary code workload is a routing stress test, not semantic retrieval

The public-query Hamming reference is 100% Top-1. Because the observation is a one-bit corruption of a known codeword and relevance is determined by exact class identity, the task has an O(1)-in-M codebook-routing shortcut using the query bit pattern and its one-bit neighbors.

Consequently, C5 results on this workload do not establish learned semantic retrieval. A later benchmark must make relevance relational or otherwise non-isomorphic to Hamming proximity.

## 5. Persistence is not measured in the main noisy funnel

The main noisy arm does not receive persistent state. The clean persistent/reset control measures the information-access control only. It does not establish persistence as an input to the noisy retrieval funnel.

These corrections do not change the registered measurements; they narrow the claims to what the numbers support.
