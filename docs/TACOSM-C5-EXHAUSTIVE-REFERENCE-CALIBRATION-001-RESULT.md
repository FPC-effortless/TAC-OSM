# TACOSM-C5-EXHAUSTIVE-REFERENCE-CALIBRATION-001 — Result

Status: MEASURED. The population-exposure confound is confirmed, but the preregistered
materiality criterion at M=512 is not met.

## Run and artifact integrity

- GitHub Actions run: 36810643404
- Full artifact: 11139324912
- Artifact SHA-256: def220ea15c44d07c1dbfc15d8d2b0f00225ef4179459a4b63a126e0b6fe5980
- Artifact size: 1,964 bytes
- Dispatch head: 0ecd5f5e16653c9defe8e959cd2bf96c3246626a
- Protocol amendment: CAL-001-A1
- Target teacher updates: 24,576; actual matched-arm updates are recorded in the artifact and remain within the registered 1% budget.

## Primary result

| M | current_48 | matched_M_minus_16 | Difference |
|---|---:|---:|---:|
| 128 | 0.386 | 0.672 | +0.286 |
| 256 | 0.166 | 0.386 | +0.220 |
| 512 | 0.053 | 0.125 | +0.072 |

The matched population materially raises exhaustive recall at M=128 and M=256 in
absolute terms, but the registered materiality test is specifically at M=512.

At M=512, the paired seed-bootstrap difference is +0.072 with 95% CI
[+0.031,+0.115]. The lower bound is below the preregistered +0.10 threshold, so
the materiality criterion is not met.

The seed-bootstrap intervals for the two pooled recalls at M=512 are:
current_48 [0.033,0.074] and matched_M_minus_16 [0.094,0.155].

## What the calibration establishes

CAL-001-A1 successfully removed the original optimizer-update confound: both arms
target approximately 24,576 teacher updates rather than allowing the matched arm
to receive a larger budget merely because its training population is larger.

The direction is consistent across the registered population levels:
0.386→0.672, 0.166→0.386, and 0.053→0.125. However, at the staked M=512 level
the measured +0.072 effect does not cross the fixed +0.10 practical-materiality
threshold.

The matched teacher also still degrades strongly with population size:
0.672→0.386→0.125. Thus population exposure explains a substantial part of the
historical exhaustive-reference collapse, but does not remove the degradation.

## Interpretation boundary

This is a calibration result only. It does not establish selective-retrieval
capability, a sparse-frontier win, semantic generalization beyond the tested
distribution, or asymptotic scaling. It does not change the registered C5 status.

## Reproduction

Full artifact:
TACOSM-C5-EXHAUSTIVE-REFERENCE-CALIBRATION-001.json

Artifact ID: 11139324912.
