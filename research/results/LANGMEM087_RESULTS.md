# LANGMEM087 stage A — longer training does not establish consistent slow-state usefulness

All six continuations, nine checkpoint evaluations and the aggregate passed.
Additional training improved next-byte prediction relative to 160 updates,
but removing older slow-state content slightly **improved** average NLL at
every tested duration. Small favorable owner and stream-position results are
preserved below. This is internal research, not independent qualification.

Only training duration changed. The original 100,521-parameter architecture,
optimizer, data sampler, 256-byte windows, gates and loss stayed unchanged.
Slow-read bias, longer-context credit assignment, stronger writes, dropout,
historical supervision and read-before-write retraining were **not run**.

## Primary results

Three original LANGMEM086 seeds continued from their exact 160-update model
and AdamW checkpoints to 640 and 1,280 updates. All nine checkpoints sealed
before either new Rust chapter was acquired. Each checkpoint predicted the
same 3,584 distinct source positions: clocks 256–2,047 in each chapter.
The 32,256 predictions across seeds and durations are repeated measurements,
not 32,256 independent text samples. NLL is in nats per byte; lower is better.

| Updates | Sampled FIT targets per seed | NLL | Top-1 accuracy | ΔNLL removing old slow content |
| --- | --- | --- | --- | --- |
| 160 | 163,840 | 2.823832 | 25.04% | -0.001585 |
| 640 | 655,360 | 2.696131 | 30.11% | -0.001454 |
| 1,280 | 1,310,720 | 2.717111 | 32.22% | -0.001769 |

ΔNLL means intervention minus full-model NLL. A positive value would favor
the original read; a negative value means removal helped. FIT counts include
repeated source exposures. All durations were preregistered; 640 updates was
not selected through early stopping. This new 160-update baseline uses a
different cohort/window from LANGMEM086, so its NLL is not directly comparable
with that experiment's 2.6879.

All three seeds improved from 160 to 640 updates, then worsened from 640 to
1,280: +0.022490, +0.015348 and +0.025102 NLL. Accuracy continued to rise.
This divergence alone does not identify a calibration or overfitting mechanism.

## Old-state versus current-write attribution

At each original prediction state, freeze the query, previous writer, value,
gates, mixture and other prior state. Decompose the slow update as:

```
S_new = H + C
H = rho * S_old * (I - eta * k * k^T)
C = eta * v * k^T
```

The old-state removal control reads `C` only, preserving the direct current
write. Current-write removal reads `H` only; this is not a read of `S_old`
before writing. The old-state permutation control replaces `S_old` with a
cyclic permutation of its value rows in `H`, while preserving `C`. All original
matrix writes and non-upper states remain bitwise unchanged. Counterfactual
upper states are discarded; every next query uses the original trajectory.
These are immediate read interventions, potentially outside the training
distribution, rather than persistent erasures or separately trained models.
Nonlinear NLL effects need not add.

| Updates | Δfast read off | Δslow read off | Δboth off | Δold slow permuted | Δcurrent slow write off | Mean slow share | Mean slow gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 160 | +0.050507 | -0.001571 | +0.057156 | +0.000106 | +0.000011 | 16.06% | 0.005363 |
| 640 | +0.177994 | -0.001470 | +0.185915 | -0.000090 | -0.000020 | 9.71% | 0.004468 |
| 1,280 | +0.290594 | -0.001803 | +0.308855 | -0.001818 | -0.000063 | 14.35% | 0.003646 |

Dependence on the fast read grew. That lesion includes its current write, so
it does not establish historical fast-byte recovery. Mean slow write gains
decreased. Mixtures and gains above are measured after training on this DEV
cohort, not illustrative equal-logit initial weights.

## Seed variation and favorable exceptions

| Seed | Updates | Full NLL | Δold slow removed | Δold slow permuted |
| --- | --- | --- | --- | --- |
| 1701 | 160 | 2.845761 | -0.003343 | +0.000344 |
| 1701 | 640 | 2.756813 | -0.002136 | +0.001091 |
| 1701 | 1,280 | 2.779303 | -0.001377 | +0.000982 |
| 1702 | 160 | 2.815620 | -0.000513 | +0.000162 |
| 1702 | 640 | 2.664074 | +0.000198 | +0.000522 |
| 1702 | 1,280 | 2.679422 | -0.000595 | -0.000763 |
| 1703 | 160 | 2.810115 | -0.000900 | -0.000187 |
| 1703 | 640 | 2.667507 | -0.002422 | -0.001883 |
| 1703 | 1,280 | 2.692609 | -0.003335 | -0.005671 |

At 1,280 updates, removing old slow content helps all three seed averages.
At 640 updates, seed 1702 has a small favorable original-read effect
(+0.000198), while the other two do not. Favorable owner cells are also retained:
640/1701/ch07 (+0.000382), 640/1702/ch09 (+0.000980), and
1,280/1701/ch07 (+0.004365). Complete owner results are in the aggregate JSON.

At 1,280 updates, the mean old-state removal delta is +0.002053 for stream
clocks 256–511, +0.000467 for 512–1,023, and −0.003843 for 1,024–2,047.
These are stream positions, not ages of retrieved content. They preserve
small positive strata without establishing long-term recall. Two source
owners and shared targets do not support broad significance claims.

## Autonomous query and simple references

| Updates | FIT-only prior NLL | 1,928-byte ring NLL | Context top-1 hits when a match exists |
| --- | --- | --- | --- |
| 160 | 3.408396 | 2.952908 | 0/52 per seed |
| 640 | 3.413822 | 2.957271 | 0/52 per seed |
| 1,280 | 3.415586 | 2.959052 | 0/52 per seed |

There were 224 diagnostic queries per checkpoint, with 52 matching-context
cases. The model received no evaluator-supplied historical address or target.
Ranking original writer keys by cosine is only a four-byte context proxy;
sign changes, transformed addresses and distributed coding limit its negative
interpretation. Zero hits do not prove absence of information. No historical
byte decoder or factual-retrieval success was established.

A separately trained fast-only comparator was not fitted. This stage therefore
does not meet all three proposed criteria for convincing slow-memory
usefulness or establish architectural superiority. The ring is a storage
reference, rather than a matched trained neural architecture.

## Sources, training and resource use

FIT used the original 14 PSF CPython owners and exact frozen source/sampler
hashes in `FIT_PLAN.json`. Original checkpoint pins are in `PARENTS.json`.
AdamW learning rate 0.001, weight decay 0.01, gradient clipping 1, and batches
of four 256-byte windows stayed unchanged. State resets at each training
window; more updates do not create a training credit horizon beyond 256 bytes.

The two new Rust Project book chapters are from commit
`2771df165f092a4fbb9f46b5b2e7450f19010adb`, licensed MIT OR Apache-2.0.
Git blob, size and license metadata were frozen before fitting. Raw content
was acquired after all checkpoints sealed; SHA256 was recorded afterward:

- `src/ch07-01-packages-and-crates.md`: 3,806 bytes; SHA256 `7141297fcc8bd0d0cb4e850b9f853574ddbb15b48a8b56c615539979c8a9e27e`.
- `src/ch09-01-unrecoverable-errors-with-panic.md`: 8,520 bytes; SHA256 `edce46b44eee961106ca49be8b1ce306e400c767304e9005b8763c141a444d1d`.

These are different chapters from LANGMEM085/086, but from the same publisher.
This is limited new-document transfer from PSF FIT, not broad independent
qualification. No post-seal SHA256 was represented as independently frozen
before fitting.

Additional fitting took 6,417.143 process-CPU seconds (106.95 minutes) across
six continuations. Evaluation took 1,041.950 seconds (17.37 minutes) across
nine checkpoints, below conservative budgets of 180 and 75 CPU minutes.
Different training batches are not a held-out learning curve.

Persistent model tensors use 1,920 bytes per owner, plus an 8-byte clock and
UTF-8 owner metadata. Shared learned weights use 402,084 bytes; the fixed code
buffer uses 8,192 bytes. The ring uses 1,928 bytes (16-byte header and
1,912-byte payload), plus a shared 2,048-byte FIT prior. Workspace is separate.
It reaches capacity late in the stream. No total-memory Pareto claim is made.
Peak evaluation RSS was 344,510,464–345,436,160 bytes.

Diagnostic-wrapper forward timing was 1.192–1.827 ms per observed byte,
including split/trace bookkeeping. Ring queries took 2.532–3.375 ms each,
excluding append. These differ in work and units; they do not establish
production latency or an equal-compute efficiency advantage.

## Verification and original evidence

Actual 100,521-parameter gates, parent/sampler/optimizer chains and all six
continuation cold logits/state/optimizer-step replays passed. Original
160-update receipts retain their original logits-replay checks. Every update
had finite connected gradients; all 35 parameter groups had nonzero finite
norms at recorded continuation endpoints. Default diagnostic predictions and
state matched the original native graph bitwise for every checkpoint.
All 18 source future-poison pairs passed. Maximum prefix error was
1.144409e-5 (limit 3e-5); split error was 1.490116e-8 (limit 1e-6).
Weights remained frozen during evaluation. Unsupported probes remain
INCONCLUSIVE.

Local verification passed 1,602 tests with four skipped after the scoped
LANGMEM086 import fix, plus 34 targeted checks after scoping both imports.
Initial import-collision CI `40983854617` and proof CI `40983914975` failures
are preserved in `ENGINEERING_LEDGER.md`. Corrected Lean/Python CI
`40984197814` and bridge CI `40984526086` passed. Actual-source Float64 split,
read and local derivative errors were 6.78e-21, 4.34e-19 and 1.74e-13.
Full neural IEEE/AD proof, semantic goals, independent reward/ACL qualification
and coordinated canonical-registry append remain OPEN.

Original fit, first-DEV and aggregate run:
https://github.com/FPC-effortless/TAC-OSM/actions/runs/40984392435

Registered source: `9d6cda3be8e200fd607bf6f8bb39fa381587bc22`.
Explicit launch: `535fbfafe6202f5a8aaf53629150372e60bd7cb1`.
Drafts: TAC #630 and PNDS #84. Aggregate artifact: `12948075104`;
ZIP SHA256: `9338ad3c168bd2aacad2b5f63883a9f67b0be8fe9167bb7ce6cf92226dd7e0d3`.
Full artifact IDs/digests and checkpoint hashes are saved beside this report.
Downloaded evaluation/aggregate ZIP hashes, all nine checkpoint files and
parent chains passed verification. Read-only reaggregation equals both the
original artifact and Actions log exactly, without further fitting, inference
or DEV-text access. See `REPRODUCE.md`. Report commits do not restart training.

## Research decision

More updates alone do not resolve consistent slow-state usefulness in this
256-byte training task. Keep the architecture and preserve the negative
1,280-update evidence. A separately registered credit-horizon and/or trainable
slow-read preference study with fresh DEV and a matched trained fast-only
control remains warranted. Do not tune on these now-exposed chapters.
This result does not rule out larger training scales, stronger task pressure,
other read/write alignment or compressed associative memory in general.
Original artifacts and failures remain intact. No GPU quota, protected TEST,
checkpoint promotion or merges were used.
