# DEV-016 — CPU-trained temporal transfer: stronger actions, failed predictive gate

**REPLAY-VALIDATED ADAPTIVE DEVELOPMENT ONLY. Preset primary success gate FAILED.**

CPU PyTorch was available through GitHub Actions. The completed pinned
[training run 38532288200](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38532288200)
passed **1,011 tests**, suite cardinality, four-arm smoke, three full training
seeds and saved-checkpoint neural evaluation replay. Training source commit:
`1b62d49db358d1e703256cfc0ab8524d3b73d118` (stacked on DEV-015).

The separate standard-library
[raw evidence audit 38533196222](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38533196222)
also succeeded. It authenticated all **12 checkpoint files**, regenerated
targets and public fingerprints from the temporal environment, derived
classifications from saved logits, checked raw outcome/verification arithmetic,
checked matched observations/source hashes across seeds, and recomputed all
**13,824 model-episode rows** and seed statistics. Neural replay and this
independent implementation of evidence arithmetic are different checks; neither
reproduces SGD or constitutes external scientific replication.

## Exact evidence

[Complete audited evidence](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38533196222/artifacts/11878714900):
`DEV-016-COMPLETE-AUDITED-EVIDENCE-NOT-CONFIRMATORY`, artifact **11878714900**,
ZIP SHA-256 `a3a7e3390839b7b07fbb0dd8a8e2fb5327164a09ff487c3d6c5d2ff9e3d0ad1a`.
Includes all three seed reports, 12 model checkpoints and `analysis.json`.

Original seed ZIPs from the training run:

| Seed | Artifact ID | ZIP SHA-256 |
|---|---|---|
| 43 | 11878766712 | `0ec8e1ebd2698ee0a31047a22c22abd5677df70ca323e308a754e7f34cf69611` |
| 59 | 11878748623 | `1114e1ea1db117e96e33fab201aa44ae1cb68499a180601c35c79599d84a9391` |
| 71 | 11878639508 | `38422a4ee51f1a23dcf8254b3835917570275860041f716c09769f8ea3bfc504` |

These immutable run/artifact IDs define the reported result, rather than any
later automatic rerun after documentation commits. Later full retraining is
restricted to explicit workflow dispatch; regression and smoke remain active.

## Architecture and measured task

Four byte-identical initialized arms inherit authenticated DEV-013
`aligned_aux` parents at seeds 43/59/71, with exactly **768 recurrent policy
scalars**. Public event/query maps feed the actual TAC-OSM delta writer,
CASM-aligned reader, sparse conditional operator execution and new output heads.
Every arm trains its evaluated future head on true labels through detached
features; predictive or shuffled auxiliary labels reach features through a
separate matched auxiliary head. Frozen writer freezes the event map too.
Therefore the joint intervention includes event-interface adaptation and shared
write/read key learning, not just isolated delta-write-gate adaptation.

Each arm gets 256 updates × eight episodes, with matched public observations
and task labels. Offline true future labels are additional supervision, not
hidden inference inputs or autonomous learning. Trainable parameters and
gradient work differ even though total architecture capacity matches.

Evaluation: three histories (32/128/512), three delays (4/16/32), four roles
(old retention, recent acquisition, cancelled old service, pending cancellation),
16 opposite-past pairs per cell. Each checkpoint sees **1,152 episodes**.
The pending condition has the same initial observed failure and shared present
query, but different historical generation-invalidating events and opposite
future outcomes. The verifier sees a public post-action inspection only;
pre-action prediction/action scores are never repaired using that receipt.

## Held-out means over three model seeds

| Arm | Action accuracy | Wait-future accuracy (primary) | H512 future | Pending-cancellation future |
|---|---:|---:|---:|---:|
| Frozen writer | 59.32% | 57.06% | 54.95% | 50.93% |
| Joint task | **73.64%** | 69.44% | **63.11%** | 55.09% |
| Joint predictive | 71.67% | **71.82%** | 62.33% | **57.64%** |
| Joint shuffled | 73.38% | 66.55% | 60.76% | 55.32% |
| Reset learned-state diagnostic | 50.00% | 50.00% | — | — |
| Hand-coded 41-scalar schema reference | 100.00% | 100.00% | 100.00% | 100.00% |

The reference is a known sufficient-statistic controller under the benchmark's
restricted schedules, not a trained competitor or a learned TAC-OSM result.
Reset uses the same trained weights, not an independently trained architecture.
Service-future labels are always zero in this evaluation and are a simple
diagnostic: predictive 99.31%, frozen 93.08%, joint task 80.47%, shuffled 76.39%.
High service-future accuracy alone does not establish a useful world model.

### Primary preselected future-state contrasts, exploratory three-seed t intervals

| Joint predictive minus control | Mean difference | 95% interval, df=2 |
|---|---:|---:|
| Frozen writer | **+14.76 pp** | **[-4.16, +33.67]** |
| Joint task | +2.37 pp | [+0.20, +4.54] |
| Joint shuffled | +5.27 pp | [-6.02, +16.55] |

The primary predictive-versus-frozen interval includes zero. The mean H512
future score, **62.33%**, also misses the preset **65%** gate. These fail the
committed success rule despite the large overall point improvement. The small
predictive-versus-task future-head advantage is consistent across these three
seeds, but extra labels/work and adaptive development limit the interpretation;
the shuffled-label contrast does not reliably establish a true-label benefit.

### Secondary native-action result and tradeoffs

Joint task minus frozen writer gives **+14.32 pp** action accuracy with an
exploratory paired-seed 95% t interval **[+8.80,+19.85]**. This is a useful
development signal for the *joint writer/interface intervention* in this new
task. It is a secondary endpoint, not a rescue of the failed predictive gate,
and it does not isolate improvement of the inherited writer alone.

Predictive versus joint task action accuracy is **-1.97 pp**
[-16.21,+12.27]. Predictive versus shuffled action is **-1.71 pp**
[-16.86,+13.45]. Adding true future supervision does not reliably improve
native maintenance decisions beyond joint task training.

Predictive supervision helps old-retention future recovery (67.82% vs 61.23%
joint task) but reduces recent acquisition (88.66% vs 91.09%). Pending
intervention dynamics remain the weakest predictive condition (57.64%).
The primary improvement is uneven across seeds: +20.31/+17.88/+6.08 pp over
frozen at 43/59/71. All seeds remain included.

Post-action failure alone is not a decision-quality metric: predictive failure
is 15.02% versus joint task 20.86%, but unnecessary service rises to **13.31%**
versus **5.50%**. Always servicing can suppress failures while violating the
benchmark's requirement to avoid unnecessary interventions. The unique optimal
action criterion and raw service rate must accompany the outcome metric.
Public verification confirms agreement with fresh observed outcomes; it does
not certify that an action was cost-optimal or repair the initial score.

## Scientific decision and next blocker

**Outcome: mixed development result; negative for the preset reliable
predictive-state gate.** End-to-end writer/interface adaptation shows stronger
native actions than the frozen-writer control on this temporal process, unlike
DEV-015's inconclusive address/XOR comparison. Nonetheless the primary future
contrast is uncertain, H512 remains weak, and the genuinely delayed pending
condition stays near 55–58%. No robust long-term memory, autonomous exploration,
full hardened loop, independent confirmation or efficiency claim is licensed.

The next controlled experiment should isolate event-map adaptation from actual
writer changes, add an independently trained reader-only temporal control,
and compare a fixed-capacity schedule-aware/schema-indexed recurrent writer.
The hand-coded reference solves the restricted task in 41 scalars, so a larger
memory budget is not required by the task itself. Its success does not prove
the current learned architecture can discover that sufficient statistic.
Introduce genuinely policy-selected sensing and unfamiliar intervention
compositions only with a frozen evaluator and appropriate finite-state controls.

DEV-015 and earlier reports remain intact. PR #172 remains draft and unmerged.
