# G-CASM-014 — Result and Scientific Disposition

**Authoritative run:** 37218553516  
**Head:** 44a2ccaef73e4a02caea83c35e7d04c22dacab39  
**Generator:** c31554413301e3c9d3e6b3f8c8c6be572a74a748  
**Repository gate:** passed  
**014 focused tests:** passed  
**Full measurement:** passed  
**Artifact:** 11310210176  
**Artifact digest:** sha256:fd65d353544f0f9b66b863cc5fdcfa849dfea2f69e1d4d63b7e199e59c306933

## Primary result

The preregistered endpoint is the seed-level paired difference in fixed-budget exact capability at K=4 probe observations and B=8 candidate executions.

| M | Adaptive success | Fixed-order success | Paired delta | Seed bootstrap 95% CI |
|---:|---:|---:|---:|---:|
| 32 | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] |
| 64 | 1.0000 | 0.98125 | +0.01875 | [0.00625, 0.03125] |
| 128 | 0.99375 | 0.73750 | +0.25625 | [0.2000, 0.3125] |
| 256 | 0.53750 | 0.50625 | +0.03125 | [-0.0500, 0.1125] |
| 512 | 0.23125 | 0.21250 | +0.01875 | [0.0000, 0.04375] |

The registered decision rule is therefore **positive at M=64 and M=128**, but it is **not positive with the preregistered interval criterion across M=256 or M=512**. The experiment does not license a general claim that adaptive probing improves capability at large M.

## Information-acquisition result

The more important diagnostic is the compatible-hypothesis count after K=4 observations.

| M | Lower collision bound M/16 | Adaptive mean | Fixed mean | Adaptive excess over bound |
|---:|---:|---:|---:|---:|
| 32 | 2.00 | 2.025 | 2.90625 | +1.25% |
| 64 | 4.00 | 4.050 | 5.20 | +1.25% |
| 128 | 8.00 | 8.050 | 10.075 | +0.625% |
| 256 | 16.00 | 15.99375 | 18.36875 | effectively at bound |
| 512 | 32.00 | 32.16875 | 39.34375 | +0.527% |

Thus the greedy minimax selector recovers almost all of the observable-partition headroom left by the frozen row order. At M=512 it reduces the mean compatible pool from 39.34 to 32.17 candidates, essentially the finite-domain collision floor.

This is stronger evidence about the acquisition mechanism than the primary success endpoint.

## K=6 and K=8 diagnostics

At M=512:

- K=6: compatible candidates 8.36 adaptive vs 11.48 fixed; B=8 success 93.75% vs 76.875%.
- K=8: compatible candidates 2.32 adaptive vs 4.66 fixed; B=8 success 99.375% vs 96.25%.

These are secondary diagnostics and do not replace the registered K=4/B=8 endpoint.

## Scientific interpretation

1. **The G-CASM-013 K=4 ambiguity diagnosis is confirmed.** A target-identity-blind adaptive row policy can nearly realize the best possible partitioning under the four-binary-row observation channel.

2. **The remaining M=512 failure is not primarily row-selection inefficiency.** With the acquisition policy already at ~32.17 candidates versus a lower bound of 32, an eight-candidate execution budget still succeeds on only 23.125% of trials. The dominant limitation is now the information content of the observation channel relative to the downstream execution budget.

3. **Adaptive acquisition has causal leverage, but its value depends strongly on the operating regime.** The largest capability effect occurs at M=128, where the K=4 adaptive partition is just above the eight-candidate execution budget. At M=512, the irreducible candidate set is much larger than B=8.

4. **The routing problem should not be described as solved in general.** What is solved here is the finite-domain *probe-ordering* problem up to the quality of the registered greedy minimax policy. Learned routing remains relevant for ranking within an ambiguity class, but a router cannot manufacture information that the four-bit observation channel does not contain.

## Important terminology correction

The M/16 quantity is not a Shannon lower bound. It is a collision/partition lower bound on expected compatible-set size for a deterministic 4-bit observation map under a uniform prior:

    E[|H(E)|] = sum_e |H_e|^2 / M >= M / 2^4.

The Shannon statement is instead:

    I(H;E) = H(E) <= 4 bits

for deterministic four-bit evidence.

## Disposition

**014 is valid confirmatory evidence.**

It supports the narrower claim that adaptive, target-identity-blind evidence placement can substantially reduce candidate ambiguity and can improve fixed-budget capability in some registered regimes.

It does **not** establish that adaptive row selection can overcome the K=4 information bottleneck at large M.

The next intervention should therefore change the **observation channel**, not merely the routing heuristic:
- structured multi-bit execution evidence;
- action-conditioned relational evidence;
- counterfactual or paired outcomes;
- cost-aware evidence selection.

G-CASM-015 has been preregistered separately for the first of these: an instrumented activation-trace channel versus scalar output observation. Its measurement is intentionally independent of the 014 confirmatory result.
