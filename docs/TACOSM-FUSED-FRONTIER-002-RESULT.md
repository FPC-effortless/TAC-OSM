# TACOSM-FUSED-FRONTIER-002 — Result

**Status: measured and reproducible.** The registered workflow passed the full repository suite and the complete measurement.

## Provenance

| field | value |
|---|---|
| branch | \`research/fused-tacosm-frontier-002\` |
| measurement commit | \`dda989ec7beb2fdf98f27341ea09873f82401eba\` |
| workflow run | 36823486572 |
| tests | 819/819 |
| artifact | TACOSM-FUSED-FRONTIER-002 |
| artifact SHA256 | \`38142658871cbf394ee0eab0159e5b2390439ab8fa24b111aec0246cd7dc8984\` |
| seeds | 10, 11, 12 |
| M | 128, 256, 512 |
| final budgets | 4, 8, 16 |
| evaluation | 80 unique-target held-out tasks per seed/M cell |

## Primary result

Frontier 001 used an exact-target endpoint that was not always identifiable. Frontier 002 corrects that by restricting evaluation to state/goal pairs for which exactly one library operator reaches the goal, and by using actual execution success as the primary endpoint.

At M=512:

| arm | B=4 | B=8 | B=16 |
|---|---:|---:|---:|
| learned PKM | 0.42% | 0.42% | 0.42% |
| deterministic structural PKM | 9.17% | 9.17% | 9.17% |
| structural PKM + StructMeans | 7.08% | 7.08% | 7.08% |
| structural PKM + StructMeans + PST | 47.50% | 68.33% | 84.58% |

At M=256, the corresponding StructMeans+PST arm was 53.33%, 65.00%, and 73.75%; at M=128 it was 78.75%, 86.25%, and 89.17%.

The deterministic structural address layer therefore restores substantial held-out admission compared with the learned candidate representation used in frontier 001.

## Admission

For the structural PKM without StructMeans, coarse admission was 100% at all three M values. This shows that the descriptor/product-key address can place the held-out target in the factorized candidate region.

Adding the StructMeans family gate lowers coarse admission to 92.08% at M=128, 81.67% at M=256, and 86.25% at M=512. The gain from the gate is not therefore established as a capability improvement; its purpose remains routing-cost reduction.

## Cost result

The cost bottleneck is now explicit.

For the structural PKM + StructMeans + PST arm, mean internal exact addresses scored were:

| M | exact addresses scored | fraction of M |
|---:|---:|---:|
| 128 | 77.89 | 60.85% |
| 256 | 163.89 | 64.02% |
| 512 | 379.20 | 74.06% |

The factor-score arithmetic was 352 measured MACs/query in the StructMeans arms, with PST prediction work of 64/128/256 for B=4/8/16.

Thus the structural representation fixes the **admission-capability failure**, but the current three-factor product-key geometry does not provide bounded addressing. The exact candidate region becomes denser as M grows.

## Execution and predictive transition learning

The unique-target construction makes execution success equal to exact target selection for this workload. This allows a direct separation:

- structural PKM: high target admission but weak final selection under the current structural score;
- StructMeans: reduces the kind search space but can discard the correct family;
- StructMeans + PST: uses the learned transition law to recover the correct candidate once it has been admitted.

The result is therefore evidence that PST is a useful **downstream compatibility scorer**, not evidence that PST solves the addressing bottleneck.

## Research conclusion

Frontier 002 supports the following narrow claim:

> A compositional operator descriptor materially improves held-out operator admission relative to the verifier-trained learned candidate representation on the registered unique-target workload.

It does **not** establish C5, bounded routing, or sublinear total addressing. The internal exact-address work remains strongly dependent on M.

The remaining bottleneck has moved from representation to **factorized index selectivity**.

## Next phase

The next experiment should keep the deterministic descriptor and PST fixed, and vary the product-key geometry rather than adding another learner:

\[
\text{StructMeans family gate}
\rightarrow
\text{N-factor structural PKM}
\rightarrow
\text{fine structural score}
\rightarrow
\text{PST}
\rightarrow
\text{AXON}
\rightarrow
\text{verify}.
\]

The immediate sweep is:

- 3 factors × 16 cells, beam 7 — current control;
- 4 factors × 16 cells, beams 4/5/7;
- 5 factors × 8 cells, beam 4.

Measure total routing work as:

\[
C_{\mathrm{route}}
=
C_{\mathrm{factor}}
+
C_{\mathrm{cell\ expansion}}
+
C_{\mathrm{exact\ address\ score}}
+
C_{\mathrm{PST}}
+
C_{\mathrm{family\ gate}}.
\]

The success condition is not simply higher accuracy. It is preservation of the StructMeans+PST capability curve while materially reducing the internal exact-address fraction as M increases.

No additional verifier, recurrence, contrastive objective, or SECA mechanism should be introduced into this routing experiment.
