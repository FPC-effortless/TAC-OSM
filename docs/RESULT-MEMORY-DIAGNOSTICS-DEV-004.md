# Memory diagnostics DEV-004: development outcome

**The evidence points toward retention/interference as a major bottleneck.
Better readout alone does not recover long histories. This is an inference
from bounded diagnostic controls, not proof that memory information is erased.**

The unchanged DEV-003 checkpoints were tested with three new diagnostic seed
streams. There were 288 retrieval pairs (576 episodes), 480 delayed-carry pairs
(960 episodes), and 1,152 independent verifier evaluation cases, plus separate
calibration cases. A 400-step readout probe was trained per frozen model using
256 separate training pairs. No parent model parameter was changed.

## Retrieval and interference

| Arm | Answer accuracy |
|---|---:|
| Native model | 56.08% |
| Exact target address supplied, oracle diagnostic | 56.08% |
| Uniform expert read, unchanged answer head | 54.69% |
| Compute bypass, unchanged answer head | 49.31% |
| Separately trained frozen-read probe | 59.03% |
| Probe with exact address, oracle diagnostic | 59.20% |
| Target moved to latest write, oracle diagnostic | 69.79% |
| Probe with target latest, oracle diagnostic | 73.96% |
| Isolated target-only memory, oracle diagnostic | 88.54% |
| Probe on isolated target-only memory, oracle diagnostic | 100.00% |
| Reset | 50.00% |

Exact-address access does not improve aggregate native performance. Target-only
and target-last interventions help much more, implicating competition and age.
The probe can perfectly decode an isolated stored bit, but improves native
full-memory accuracy by only 2.95 points. The exploratory three-seed 95% interval
for that improvement is [-1.59,+7.50] points, so the benefit is uncertain.

The oracle interventions remove or reorder evidence using evaluator knowledge;
their high scores are not operational model capability. Compute bypass and
uniform reading shift the distribution seen by the fixed answer head. Their
poor scores do not independently establish architectural necessity.

| History length | Native | Readout probe |
|---|---:|---:|
| 32 | 61.46% | 68.23% |
| 128 | 54.69% | 56.77% |
| 512 | 52.08% | 52.08% |

For targets aged 128–543 writes, both native and probe accuracy were 50%.
The median counterfactual state contrast fell from about 0.715 for targets aged
0–15 to 0.030 for targets aged 128–543; the corresponding read contrast fell
from 0.681 to 0.010. Nevertheless, **none of the paired states were exactly
equal**. There is no information-theoretic erasure proof. A stronger decoder,
different training or improved memory dynamics could still recover information.

The probe was trained on H=32 and evaluated at longer H, so its failure also
contains distribution shift. These results prioritize memory retention and
interference handling but do not fully separate capacity from optimization.

## Genuine writes after a verified commit

| New intervening writes | Unrelated writes | Near-address competing writes |
|---|---:|---:|
| 0 | 76.04% | 79.17% |
| 1 | 69.79% | 71.88% |
| 4 | 68.75% | 57.29% |
| 16 | 59.38% | 48.96% |
| 32 | 64.58% | 50.00% |

Reset and no-commit were exactly 50% in every cell. The short carry benefit does
not survive strong near-address competition at 16/32 writes. Different cells
use independent examples, so these are not matched longitudinal trajectories
or a perfectly monotone delay curve. The two zero-delay columns differ because
they sample different examples; no interfering write occurs in either cell.
Swapped accuracy is algebraically complementary on the paired labels, not
independent statistical evidence.

## Verifier calibration tradeoff

| Policy on the new evaluation distribution | False accepts | False rejects |
|---|---:|---:|
| Original neural threshold 0.6 | 19.27% | 0.00% |
| Separately calibrated threshold 0.9 | 1.91% | 14.58% |
| Public evidence-sign rule | 0.00% | 0.00% |

All three calibration streams selected 0.9 from the frozen grid, minimizing
`5*false_accept_rate + false_reject_rate` before evaluation. Threshold 0.9 is
reported as a diagnostic result and is not silently installed in the parent
model. It trades erroneous acceptance for lost valid writes. The perfect sign
rule follows from this synthetic validity definition and is not a general
verification system. This test includes ±0.5 evidence margins absent from the
previous narrow pilot; 19.27% is not directly comparable to its 13.89% result.

## Validation and development status

All 911 repository tests passed. Saved parent/probe checkpoints replayed every
retrieval, delayed-carry and verifier/calibration row exactly. Source/checkpoint
hashes, schedules, episode identities and probe/evaluation separation passed.
The original checkpoints stayed immutable; the raw artifact preserves all probe
loss steps and records. Replay uses the same diagnostic implementation and is
not an external scientific audit or independent retraining of the probes.

The protocol was frozen in local git before measurement, initially at
`f669924` and then moved byte-for-byte to `development_protocols/` at `05bcabb`
to respect the repository's confirmatory schema. Implementation was committed
at `1ce0f00` before measurement. Remote publication occurs afterward; no claim
of prospective public preregistration is made. Raw source hashes pin the exact
measured code even if remote publishing produces different commit IDs.

This remains development evidence from the existing generator family. A fresh
prospective benchmark with untouched key/corruption families and matched modern
baselines is still necessary for confirmatory architectural conclusions.

The next architectural experiment should test retention protection through
competing writes and separate it from decoder training. It should preserve
these controls and explicitly account for verifier false rejection as well as
false acceptance. Broadening the operator bank alone is not supported as the
primary remedy by these diagnostics.
