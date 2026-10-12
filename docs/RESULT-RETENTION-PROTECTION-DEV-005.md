# Retention protection DEV-005: no demonstrated architectural benefit

**The learned full-transition protection gate did not outperform the ungated
control under this fixed development budget. Keep the implementation as an
experimental arm, not a validated upgrade.**

Nine models were trained: three matched seeds for each of learned protection,
always-update and fixed sigmoid(2) protection. Each received 240 identical
training batches of eight episodes, balanced between earlier-memory retention
and acquisition of new facts. Initial parameter tensors were identical across
arms within each seed. All evaluation examples were shared across arms.

There were 432 distinct paired evaluation rows / 864 episodes per arm, with
half retention and half acquisition. Across three arms this is 2,592 model
episode evaluations; repeated arms are not additional independent data/seeds.

## Outcomes

| Arm | Overall | Retention | Fresh acquisition |
|---|---:|---:|---:|
| Learned protection | 56.94% | 53.94% | 59.95% |
| Always update | 57.52% | 54.17% | 60.88% |
| Fixed protection | 55.67% | 52.78% | 58.56% |
| Dense Euclidean matching | 83.80% | 81.48% | 86.11% |

The Euclidean comparator retains/searches all observations. It is not a
memory-budget, parameter-budget or hardware-cost matched architecture.
This generator includes fresh-acquisition queries and longer training histories;
these scores must not be directly compared with earlier DEV-003/004 aggregates.

| Model seed | Learned | Always update | Fixed |
|---|---:|---:|---:|
| 43 | 58.68% | 63.19% | 54.86% |
| 59 | 55.21% | 53.47% | 53.13% |
| 71 | 56.94% | 55.90% | 59.03% |

The paired three-seed learned-minus-always overall gap is -0.58 percentage
points, exploratory 95% t interval [-9.09,+7.93]. Learned-minus-fixed is +1.27
points, interval [-6.26,+8.81]. These intervals use df=2 and only three paired
seeds, assume approximately normal seed effects and include no benefit. They
do not prove equivalence or universal failure, and no arm/checkpoint was chosen
from its evaluation outcome.

## What the gate learned

The learned gate averaged 0.91051 during initial history and 0.91048 during new
writes. Its new-write average was 0.91069 for unrelated writes and 0.91027 for
near-address competitors. It increased update strength from its initial 0.88080
toward the always-update control rather than showing strong mean selective
suppression under this interference manipulation. Aggregate means alone do
not prove that all individual gates were constant; raw per-write mean/min/max
traces are preserved and replayed.

At H=512, learned retention was 51.39% and acquisition 55.56%; always-update was
52.78% and 62.50%. After 32 near-address competing writes, learned and ungated
retention both reached exactly 50%, while fixed protection reached 51.39%.
The earlier long-history weakness is unresolved in this new dual-task setting.

Reset was exactly 50% throughout. The no-intervening-write oracle control was
exactly 50% on every acquisition pair, validating the anti-freeze test. That
control uses evaluator knowledge of the history boundary and is not a runtime
policy. Swapped-history accuracy is complementary by paired construction, not
independent statistical confirmation.

## Architecture and budget boundaries

All arms have 37,066 total parameters and a 768-scalar recurrent state. Learned
protection actively trains 36,521 parameters; controls train 35,592. The 929
gate parameters are frozen/unused in controls, and the 545 verifier parameters
are frozen/unused in all arms. The extra learned gate computation and active
parameters mean this is not an active-parameter/FLOP matched SOTA comparison.

This gate scales the complete proposed transition: decay, erase and new-value
write. It therefore couples preserving old contents to weakening incoming
updates. A future design should test more targeted protection or separate
retention/read-write mechanisms while preserving the acquisition countertest.
That is a research direction, not a demonstrated fix. The present result does
not establish the causal reason for failure or rule out optimization/budget
limitations, representational capacity or query ambiguity under near collisions.

## Integrity and provenance

All 928 local repository tests passed. Checkpoint replay reloaded all nine
models and exactly regenerated every evaluation prediction, control, gate trace,
state contrast and aggregate. The validator reconstructed initial model hashes
and every training data fingerprint, checked cross-arm data identity and
train/evaluation separation, and enforced paired ceilings and trace/state sizes.
It does not independently retrain weights or reconstruct recorded gradient
histories; those diagnostics are tested and saved by the runner.

The original development design was frozen locally at `cf3ac5f`, amended to add
the fixed-strength control at `ec3e59a`, and implemented at `e1ca8e5`, all before
the new training/measurement. No outcomes informed those changes. Remote
publication follows local measurement; this is not public scientific
preregistration. Exact source hashes identify the measured source if publishing
produces different commit IDs. The same development-family distributions remain
in use; no untouched confirmatory or general multimodal/verifier claim follows.

All earlier experiments, weights and results remain unchanged. Preserve this
negative result before proposing another architecture.
