# Integrated memory DEV-003: measured development outcome

**Development only. Integrated historical retrieval remains unsuccessful relative
to its classical comparator. No state-of-the-art or confirmatory claim.**

Run on 10 October 2026 using Python 3.12 and CPU PyTorch 2.14.1. Three trained
seeds, 180 steps each, 144 paired evaluation rows, 288 initial-task episodes
and 288 short post-commit follow-up episodes. No checkpoint selection or
post-measurement training-budget changes were made.

| Endpoint | Result |
|---|---:|
| Learned historical answer accuracy | 54.86% |
| Dense Euclidean answer accuracy | 86.81% |
| Nonlearned 64-slot cache answer accuracy | 59.38% |
| Learned no-intervening-write accuracy | 59.03% |
| Swapped-history answer accuracy | 45.14% |
| Reset answer accuracy | 50.00% |
| Short verified-write follow-up accuracy | 70.83% |
| Follow-up no-commit / reset accuracy | 50.00% / 50.00% |
| Follow-up swapped-state accuracy | 29.17% |
| Euclidean target recall@1 | 73.61% |
| Recurrent state size | 768 scalars per episode |

The follow-up supports a bounded development observation that a newly committed
write influences the later answer. It does not establish long-delay retention
through new competing writes, generic correctness verification, or causal
consolidation in the full multimodal PLM. The learned historical model remains
31.94 percentage points below Euclidean and below the nonlearned cache.

## Seed-level outcomes

| Training / evaluation seed | Historical learned | Euclidean | Follow-up carry |
|---|---:|---:|---:|
| 43 / 131 | 55.21% | 89.58% | 75.00% |
| 59 / 149 | 55.21% | 85.42% | 65.63% |
| 71 / 167 | 54.17% | 85.42% | 71.88% |

Exploratory 95% Student-t intervals over the three paired seed differences:
follow-up carry minus no-commit = +20.83 points, interval [+8.98,+32.69];
historical learned minus Euclidean = -31.94 points, interval [-37.33,-26.56].
These use df=2 and `mean ± t(0.975,2)*sample_sd/sqrt(3)`. They assume independent
approximately normally distributed seed effects, are based on only three seed
pairs and are descriptive development intervals, not a confirmatory decision.
Episode pairs are not counted as independent extra seeds.

## Verification and work

Positive sensor evidence led to 287/288 accepted commits. Negative sensor
evidence still caused **40/288 false accepted commits (13.89%)**. Verifier
calibration is therefore a remaining failure, even though the kernel preserves
state exactly whenever its gate actually rejects the write. Do not conflate
correct rejection semantics with reliable learned rejection.

The learned first-query path dispatched 686 operator/sample calls across all
288 episodes, against 864 for always executing one selected operator at all
three depths. This establishes counted dispatch behavior only; it is not a
FLOP, latency or quality-matched efficiency result. The fixed-state memory still
ingests every history/write observation. No sublinear end-to-end claim follows.

## Evidence validation

The validator reloaded all three saved checkpoints, regenerated all 144 pairs,
and exactly reproduced raw predictions, controls and write/dispatch traces.
It checked source hashes, checkpoint hashes, complete evaluation scheduling,
episode uniqueness and train/evaluation fingerprint separation. Aggregate and
per-seed summaries agreed. Status: `REPLAY_VALIDATED_DEVELOPMENT_ONLY`.

The local run preceded the implementation commit. Its `provenance.commit`
records the parent #158 commit; the complete source-file hashes identify the
actual development source used. This is explicit local-development provenance,
not an assertion that the parent commit contains this successor. CI runs record
the successor's head commit and preserve their own artifact separately.

Training gradient diagnostics are captured by the runner and checked by focused
mechanism tests. Checkpoint replay does not independently rerun training or
prove every recorded training diagnostic. Evaluation comes from the same
development generator family used during design. Raw checkpoints and complete
episode evidence are preserved as a separate archive and by the CI artifact.

Next research targets are historical retrieval under fixed capacity, verifier
false acceptance, longer post-commit delays, and parameter/state/budget-matched
modern memory baselines. Preserve this negative/partial result while designing
a new prospective experiment.
