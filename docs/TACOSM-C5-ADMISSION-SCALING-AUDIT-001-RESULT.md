# TACOSM-C5-ADMISSION-SCALING-AUDIT-001

**Status: REGISTERED**

## Why this phase exists

The preceding noisy full-loop result conflated two different failure quantities:

- dense-CDL rank growth, which increases shortlist size and therefore ranking/execution work;
- LSH admission loss, which can remove a target even when the dense rank is inside K90.

This audit measures those quantities separately before any representation intervention.

## Registered measurements

### Dense rank scaling

Evaluate the same trained noisy CDL router at:

`M = 64, 128, 256, 512, 1024, 2048, 4096, 8192`.

Report mean best-valid rank, P90 best-valid rank, Top-1 validity, the fitted full-range power exponent, and the local P90 exponent at each doubling.

The local exponent is:

`gamma_local = log(P90_{2M}/P90_M) / log(2)`.

### LSH table sweep

At `M=1024`, derive `K90` only from the calibration split. Measure the fixed CDL representation through one OR-LSH family while varying:

`L = 1,2,4,8,10,12,16,24,32,48,64`.

The sweep cap is 128, so no requested L is cap-bound.

For each L report admission recall, reranking count/fraction, hash arithmetic, and total routing arithmetic.

Also report the calibration-derived iid diagnostic:

`L90 = ceil(log(0.10) / log(1 - p1^b))`.

This is an approximation under independent table events, not a guarantee.

## Scientific corrections carried forward

The one-bit binary workload is exactly recoverable by a public-query Hamming reference. It is therefore a routing stress test, not evidence of learned semantic retrieval.

The main arm contains no persistent-state input. The clean persistent control from the previous phase remains a separate control and does not establish persistence inside the noisy funnel.

## Decision boundary

This audit does not authorize a representation change. First inspect the LSH recall/work curve and the high-M local rank exponents.

A later representation phase may proceed only after this audit has identified whether the immediate bottleneck is representation rank growth, LSH admission, or both.
