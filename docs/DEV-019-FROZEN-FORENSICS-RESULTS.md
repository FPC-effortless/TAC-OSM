# DEV-019 — Archived DEV-017 prediction-collapse findings

**POST-HOC READ-ONLY ANALYSIS OF FROZEN EVALUATION OUTPUTS, NOT CONFIRMATORY.**
The original DEV-017 reliability/cost gates remain FAILED. No old result or
checkpoint is changed; keep this stacked research draft unmerged.

## Provenance

- Original input: audited DEV-017 archive from [run 38551851130](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38551851130),
  artifact 11886684731, ZIP SHA-256
  9fad2ff62c511a6dc829808823a4706615906dd923ab1552395dc4ffd95983b1.
- New successful read-only forensic [run 38814616641](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38814616641),
  [artifact 12009802732](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38814616641/artifacts/12009802732),
  evidence ZIP digest
  f6603fd30339a5fa9ee16cbc2cdb9ab5127970938574db014592a837932079ba.
- All three full seed reports (43, 59, 71) were read. The tool checked 1,728
  evaluation rows per arm/seed, original stored argmax/logits, matched public
  history fingerprints across arms, original forecast-accuracy summaries,
  exactly 128 critical decisions / 64 opposite-outcome pairs per arm and seed,
  and cost arithmetic for every stored physical trajectory.
- Critical endpoint is DEV-017 H512 pending/reissued cancellation, at/after due,
  delays 64/128, not the different frozen DEV-018 primary endpoint.

## New diagnostic observation: zero counterfactual class discrimination by TAC

| Arm | Seed 43: accuracy / paired-class difference | Seed 59 | Seed 71 |
|---|---|---|---|
| schema41 | 71.88% / 43.75% | 80.47% / 60.94% | 75.78% / 51.56% |
| schema41 frozen GRU | 69.53% / 39.06% | 79.69% / 59.38% | 72.66% / 45.31% |
| tac_reader | 50.00% / **0%** | 50.00% / **0%** | 50.00% / **0%** |
| tac_encoder | 50.00% / **0%** | 50.00% / **0%** | 50.00% / **0%** |
| tac_memory | 50.00% / **0%** | 50.00% / **0%** | 50.00% / **0%** |
| tac_joint | 50.00% / **0%** | 50.00% / **0%** | 50.00% / **0%** |

There are 768 critical paired comparisons for the TAC arms
(64 pairs × 3 seeds × 4 arms). Every TAC arm assigns the **same predicted
class within each opposite-outcome pair**. Because the true labels are
balanced and opposite in each pair, this directly explains 50% accuracy:
one partner is right and the other wrong.

This does **not** imply a single constant predicted class across the entire
dataset. TAC predicted-positive rates differ markedly between arms and seeds;
the failure is **within-pair** discrimination, not necessarily global class
collapse.

The mean within-pair absolute predicted-positive probability gap (across seed
means) was approximately **0.5055 for schema41** versus **0.0000120 for
tac_joint**. For TAC encoder-only, the largest per-seed mean gap was
approximately **0.000568**. Archived logits therefore contain little
decision-relevant separation on these critical pairs, even before argmax.

The 41-scalar system itself remains imperfect. Schema41 critical-pair
class disagreement averaged 52.08% across three seeds, while its frozen-GRU
control averaged 47.92%. DEV-017's original preset per-seed success rule
still failed.

## What the result licenses — and does not

**Supported:** at the archived learned prediction endpoint, TAC's reader and
head fail to express changes that track the decisive historical cancellation
information. The same-class property is established on all critical pairs.

**Not supported:** whether that information was erased from recurrent memory,
survived but could not be read, or was never encoded. We also cannot separate
memory interference from H32/H128 training versus H512 testing. Original
low-failure, high-service policy costs and all prior negative gates stand.

No checkpoint was retrained. Raw evidence replay and physical arithmetic are
integrity checks, not independent scientific replication.

## Next proposed prospective study

Freeze DEV-017 writer weights and evaluate the *same public opposite-past
histories* at multiple prefix lengths. Measure (1) recurrent-state distance
between paired worlds, (2) same-query read-feature distance, and (3) forecast
logit distance, before and after matched unrelated writes. Train a separately
controlled post-hoc decoder probe only on disjoint training histories, with
shuffled-label, frozen-random-writer and reset-state controls. Reproduce results
on held-out paired sequences with equal readout budgets.

If frozen states distinguish the pairs while the trained predictor does not,
prioritize reader/optimization alignment. If states become indistinguishable
after distractor writes, prioritize entity addressing, interference, decay and
overwrite semantics. Only then preregister new entity-count sweeps and an
entity-factorized TAC writer comparison. Do not change DEV-018's frozen primary.
