# DEV-029: audited validation pilot and integration checkpoints

This is a six-arm engineering pilot, not the completed optimization study.
One initialization per arm, LR0.002, H512, B2, 216 updates: 432 episodes,
221,184 event observations and 432 training inspection receipts. All arms used
identical public training histories, receipts and validation cases. There was
no outcome-based pruning, recipe selection, test evaluation or branch merge.

## Evidence and reproducibility

- Draft [PR #184](https://github.com/FPC-effortless/TAC-OSM/pull/184).
- Successful [Actions run #38935456485](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38935456485): regression/integration, six training jobs and independent audit all passed.
- Protocol/implementation frozen in `1ab9abc6864bbae41ae5174702e25acf1f42a203`; actual PR merge checkout recorded by every checkpoint: `de7335e84e73e45667148896813e018a91ce14b8`.
- Complete audited artifact `12061900088`, `DEV-029-AUDITED-PILOT-VALIDATION-ONLY`: 21,656,534 bytes. ZIP SHA256 `57cd77e052900bc0d44684eddb9f34d9ac910ac979e7b4cf9230c0961b8d0ab0`.
- Archive includes all six trained model/optimizer checkpoints, raw predictions, public-stream fingerprints, physical decisions/receipts/costs and audit summary.
- Separate local checkpoint prediction/receipt/physical-cost replay passed. Its summary is exactly equal to the archived GitHub summary.
- A supplemental post-run integrity audit independently recomputed all **13,824 hypothetical inspection receipts**, their expected costs/gains and selected entities. All selected entities matched exactly; maximum gain discrepancy was 4.99e-7 or less. This strengthened integrity coverage without changing any metric or threshold. See [audit record](DEV-029-SENSING-CHOICE-AUDIT.json).
- CI at the frozen pilot source passed **1,127 tests**. The subsequent resumable-study runner integrity test increased the local suite to **1,128 passing tests**. These are distinct validation checkpoints.

The independent audit verifies registered condition coverage, identical and
nonoverlapping training/validation fingerprints, checkpoint/model hashes,
recorded source, labels, checkpoint logits within frozen inference tolerances,
exact categorical predictions and physical decisions, receipt values, and
physical cost replay. Inference tolerance is rtol1e-5/atol1e-6; categorical and
cost checks remain exact. It does not establish scientific generalization or
certify a full tuning matrix.

## Accuracy: preserve the metric distinction

DEV-029 prospectively registered an **all-role mature target** accuracy for
validation recipe selection. This includes retention and acquisition as well
as the two cancellation roles. Earlier DEV experiments' primary accuracy
uses **mature cancellation roles only**. These are different denominators;
the former must not be presented as a replication of the latter. Both appear
below, without changing either definition after seeing outcomes.

| Arm | Registered all-role accuracy (96 cases) | Earlier cancellation metric (48 cases) | Parameters | Training CPU seconds |
|---|---:|---:|---:|---:|
| Native TAC | 52.08% | 50.00% | 8,674 | 97.14 |
| Masked-local TAC | 87.50% | 75.00% | 8,674 | 94.39 |
| Learned schema41 | 100.00% | 100.00% | 5,932 | 60.79 |
| GRU768 | 55.21% | 50.00% | 1,831,714 | 88.02 |
| Gated delta | 50.00% | 50.00% | 7,146 | 64.65 |
| Classical table + learned head | 75.00% | 50.00% | 482 | 9.75 |

Times are observations from separate CI jobs, not paired hardware-normalized
speed comparisons. Equal updates/search opportunities do not equal FLOPs.
The table and schema recurrent states occupy 164 float32 bytes, other arms
3,072; parameter, optimizer, temporary activations, routing and decision costs
are additional. Measured full FLOPs are still missing. Validation wall times (including public
encoding, prediction and hypothetical sensing) were 9.47, 7.30, 4.92, 7.56,
6.36 and 3.31 seconds in table order. GRU's model/optimizer checkpoint occupied
21,987,093 bytes; every other arm's was below 131 KB. These runtime/archive
costs are retained alongside recurrent state size.

The classical table's state is a task-informed sufficient statistic; its head
receives raw public state and query features, not an oracle forecast. Despite
that sufficient state, its small head did not solve the cancellation distinction
in this limited recipe. This reinforces the need to separate representation,
feature conditioning and optimization. It does not prove that more updates
alone will solve it. Schema41's supplied entity slots are not learned discovery.

## Economic utility: absolute outcomes and small differences

These totals cover the **144 validation episodes** (1,152 entity decisions),
including before-deadline conditions. Costs include failure4, service1 and
inspection0.1; means are per entity. They are descriptive episode measurements,
not twenty-seed uncertainty estimates.

| Arm, selected policy | Failures | Unnecessary services | Inspections | Mean cost/entity | Selected minus model-only |
|---|---:|---:|---:|---:|---:|
| Native TAC | 0 | 562 | 0 | 1.000000 | 0 |
| Masked-local TAC | 0 | 24 | 34 | 0.535938 | −0.003125 |
| Learned schema41 | 12 | 0 | 1 | 0.543490 | +0.000086806 |
| GRU768 | 0 | 562 | 0 | 1.000000 | 0 |
| Gated delta | 0 | 562 | 0 | 1.000000 | 0 |
| Classical table + learned head | 0 | 24 | 0 | 0.532986 | 0 |

Zero failures for native/GRU/delta come from servicing every entity, with
substantial unnecessary work and cost. Conversely, perfect cancellation
accuracy is not perfect performance on all other conditions: schema41 has
12 failures across the full physical evaluation.

Schema41's selected-cost comparison fails the preserved strict `selected <=
model-only` component: its single inspection adds 0.1 total cost, or approximately
0.000086806 per entity. Keep that exact failure, while reporting its small
magnitude. It does not support treating the model as broadly economically poor.
Masked-local's cancellation score is below the historical 80% floor. The other
four arms also miss that floor. Thus no arm could pass every original combined
criterion on these pilot measurements; full attribution/reliability gates have
not been assessed, and this is not a twenty-seed system score.

[Descriptive summary](DEV-029-PILOT-DESCRIPTIVE-SUMMARY.json) includes per-policy
cost ranges, medians and quartiles, absolute outcomes, and raw-file hashes.
[Independent summary](DEV-029-AUDITED-PILOT-SUMMARY.json) retains the frozen
pilot's original metric and cost totals. No seed-level confidence interval is
estimated from a single initialization or from correlated episode counts.

## Complete-system checkpoint, kept separate

The preserved `FullLearnedPLM` reference executed representation, multi-timescale
MTSK, learned routing, learned CASM, prospective OSM action scoring, verifier,
repair proposal and verifier-gated state commit on random-weight synthetic
inputs. Outputs were finite and action scores had nonzero CASM/OSM gradients.
This is wiring evidence, not trained capability. No physical environment or
repair execution was included in that checkpoint; the learned router is not
the original NLL CDL teacher.

Separately, the pinned PR #146 fusion checkout completed 24 existing synthetic
physical-environment steps. It exercised sparse routing, specialist-driven
action selection, multi-timescale memory, regime updates, verification,
repair proposals and 17 gated experience writes. It did not call prospective
OSM or execute proposed repairs. Its source remains
`5923ed0bf1aa2b9ae125c98d60dab25ec3cf07aa`; nothing was merged.
See [fusion boundary record](DEV-029-FUSION-BOUNDARY-CHECK.json).

Neither checkpoint supplies evidence that all original components together
outperform their ablations. Keep graph CASM, neural conditional operators and
fused specialists as distinct candidates until explicit interface and causal
contribution comparisons are registered.

## Next controlled milestone

Finish all registered tuning trials and nested 216/648/2160 checkpoints, audit
complete coverage, freeze validation-selected recipes in a separate commit,
then run twenty fresh initializations and untouched test cases per arm. The
resumable tuning runner is prepared and its optimizer-state continuation is
verified on a short independent integrity fixture. Full-matrix auditing and
the post-freeze test evaluator still need implementation. No test access is
unlocked by this pilot or by the selector alone.

DEV-028 remains deferred. Indirect identifiable identities, 8→16/32 capacity,
transition-rule generalization, original CDL supervision, executed repair and
end-to-end component ablations remain first-class unfinished research goals.

### Resource planning for the next batch

The six pilot arms consumed approximately 415 aggregate single-thread training
CPU seconds. Linear extrapolation to all 72 nested 2,160-update tuning trials
is about **14 CPU hours of training**, before curriculum effects, validation,
serialization and job setup. This is a rough planning estimate, not a measured
study cost or an architecture-efficiency result. The shorter-history curriculum
will change the estimate. Twenty-seed reliability is additional. The full grid
was not launched under this batch's approximately 30-minute allowance; future
batches must retain all registered cells and mark the matrix incomplete until
finished rather than promote whichever trials finish first.
