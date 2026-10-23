# DIAGKEY004 — original synthetic-only results

**Original training and scoring complete; independent clean-checkout replay pending.**
No real-world capability or full-compute-matched architectural advantage is established.

Preregistered code/plan: `79b11a5e`. All25 final models and50 original640/4096-update
checkpoint+optimizer/RNG files were frozen at `04002ef6` BEFORE the first TEST query.
Five seeds, four distinct random key/value stores then one answer-free QUERY,
zero distractors,32 values/unconditional chance3.125%,512 synthetic queries per seed per model.
An answer chosen uniformly from the four observed values has25% expected accuracy
(and NLL log4) without using the query key; GRU/no-association≈24% does not establish
reliable key binding.
The same2,560 synthetic queries are reused across arms/milestones;50 score sets
are NOT25,600 independent cases. All original logits/labels/correct bits are stored.

| Model | Parameters | Mean accuracy,640 updates | Mean accuracy,4096 updates | Lowest final seed | All5 >=90% |
| --- | ---: | ---: | ---: | ---: | --- |
| Aligned native TAC | 101,527 | 97.93% | 99.84% | 99.41% | PASS |
| Unchanged native TAC | 101,554 | 30.31% | 94.02% | 77.15% | FAIL |
| No-association TAC | 101,560 | 17.93% | 24.26% | 22.46% | FAIL |
| GRU | 101,500 | 19.02% | 24.18% | 21.48% | FAIL |
| Causal Transformer | 101,616 | 100.00% | 99.92% | 99.61% | PASS |

The revised model already passes every seed at640 updates (97.07–98.63%).
Final revised seed accuracies are99.80%,100%,99.41%,100%,100%. The final registered
gate passes unchanged. The original native also learns much better with more
updates:30.31% ->94.02% mean, yet seed75001 remains77.15%. Thus the earlier
640-update failure partly reflects a budget/recipe limitation; it did not prove
inherent inability. GRU/no-association controls remain near24% under this recipe;
do not generalize this into a claim that GRUs cannot learn immediate recall.
The Transformer positive control passes every seed at both checkpoints.

| Model | Mean NLL at640 | Mean NLL at4096 | Mean fitting process CPU seconds |
| --- | ---: | ---: | ---: |
| Aligned native TAC | 0.126466 | 0.005588 | 28.02 |
| Unchanged native TAC | 2.240917 | 0.165050 | 27.68 |
| No-association TAC | 2.663053 | 1.701398 | 18.95 |
| GRU | 2.603520 | 1.685433 | 16.55 |
| Causal Transformer | 0.001120 | 0.018144 | 21.84 |

Fitting CPU includes the registered learning-curve DEV checks and checkpoint
writes; it is not inference latency or an isolated kernel benchmark. More
training is not uniformly better: the Transformer goes from100% to99.92%
and its final mean NLL worsens, dominated by seed75001. All preregistered
milestones/seeds remain visible; the final checkpoint was not reselected.

## Architecture and scope

The candidate retains88D fast and88D slow recurrent states and ONE8x8 memory.
Changes are shared learned normalized addresses, tied learned value-code
writing/decoding, separate retention/error-write gates, exact read-only QUERY
and radius-four memory projection. All modules receive finite gradients; the
entire101,527-parameter model is guarded before updates and at serialization.
No hardcoded first-four-step write mask or answer-bearing query is used.
The address equality is by construction, not evidence of learned semantic
alignment. Individual effects are not isolated. All math/source/registration
hashes stayed frozen during fitting. No delay training or LR sweep occurred.

## Fairness and negative evidence

All arms share examples/order, hyperparameters and640/4096 milestones, with
parameter discrepancy<0.12%. **Full training-FLOP parity is unverified.**
Partial supported forward/backward estimates, versus old TAC: aligned−4.15%,
no-association−8.01%, GRU+1.51%, Transformer+3.12%. These signed gaps are
implementation measurements, not complete operations. Do not claim a fair
architecture winner, superior sample efficiency in general, or real-world gain.
CPU equality remains unnecessary; no dummy operations or extra exposure added.

## Frozen DIAGKEY003 read check

An additional no-fit check on each old original checkpoint reproduces its
full forward logits, then compares original versus oracle-key reads through
the unchanged decoder, and FOUR-way latent write-value identification.
The four-way metric has25% chance; the actual task has32 classes.
Three seeds show target/query cosine−0.87 to−0.91 while their old TEST accuracy
is32.6–35.2%. Oracle-key substitution often worsens their32-way predictions.
This does not prove retention perfect or retrieval uniquely broken: coding
conventions and OOD readout substitutions matter. See the full per-seed record
`FROZEN_DIAGKEY003_READ_CHECK.json` and `PRIOR_ART.md`.

## Decision

**PASS the bounded immediate-recall mechanism gate.** The capability scorecard
remains ENGINEERING_VERIFIED for synthetic evidence; no authentic case has
been scored by the new model. Long-delay interference/relevance, long-memory
retention, missing observations, real enterprise/physical transfer and action
utility remain unknown. Consider a separately preregistered interference
curriculum; do not infer success or silently train the next ladder stage.
Keep this candidate in a draft branch; existing real models/holdouts unchanged.
