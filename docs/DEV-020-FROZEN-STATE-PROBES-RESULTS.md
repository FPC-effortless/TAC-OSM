# DEV-020 frozen state probes — audited results, October 16, 2026

**Engineering execution and audit passed. TAC diagnostic recovery gates failed.**
This is an adaptive, synthetic, post-hoc experiment using three training seeds;
it does not establish a new native TAC capability or identify a unique failure
mechanism. The [fixed protocol](DEV-020-FROZEN-STATE-PROBES.md) remains the
authority for the measurement recipe.

## Evidence and verification

- Completed [run 38832908326](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38832908326),
  source `d5ca8b69a031ca918ccf7b8413d0d9e5957e607d`.
- All six seed/regime jobs and the independent stdlib audit succeeded. Five
  frozen architectures per job give 30 checkpoint analyses, each with 96
  held-out episodes and 288 disjoint-domain probe-training episodes. Prefix
  measurements reuse the same histories; these are not independent replications.
- [Audited evidence artifact 12017508050](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38832908326/artifacts/12017508050),
  `DEV-020-COMPLETE-AUDITED-DIAGNOSTIC-NOT-CONFIRMATORY`, contains six
  `analysis.json` files and `summary.json`.
- Downloaded archive SHA-256:
  `575dc4375bc1bc641825daa37405b22b7e8fec50fdb4a6502c77cd79d7295402`.
  Local re-execution of `audit_dev020_state_probes.py` on those six files
  reproduced the archived summary byte-for-byte. The audit validates summary
  integrity and aggregation; it does not independently retrain probes or
  reconstruct model features.
- Frozen DEV-018 checkpoints came from original run 38802569971. Per-checkpoint
  archive hashes and parameter hashes, public-history fingerprints, labels,
  archived decisions and primary accuracy were checked. Recomputed probabilities
  must match within 2e-5; archived logits/decisions remain authoritative at H512,
  so near-tie recomputation cannot rewrite DEV-018.

## Main outcomes

Percentages below are means over seeds 43, 59 and 71. Probes are separately
trained supervised diagnostics; their accuracy is not native model accuracy.
The diagnostic gate requires **every seed** to score at least 75% and exceed
its shuffled-label control by at least 10 percentage points.

| Architecture | Training | Original H512 | State probe | Read-input probe | State/read diagnostic gates |
|---|---|---:|---:|---:|---|
| schema41 | Short | 96.88% | 100.00% | 100.00% | Pass / Pass |
| schema41 | Long | 90.28% | 100.00% | 100.00% | Pass / Pass |
| tac_joint | Short | 50.00% | 50.35% | 50.00% | Fail / Fail |
| tac_joint | Long | 50.00% | 50.00% | 49.65% | Fail / Fail |
| tac_encoder | Short | 50.00% | 50.35% | 50.00% | Fail / Fail |
| tac_encoder | Long | 50.00% | 51.04% | 49.65% | Fail / Fail |
| GRU768 | Short | 50.00% | 50.00% | 50.00% | Fail / Fail |
| GRU768 | Long | 50.00% | 50.00% | 50.00% | Fail / Fail |
| Memoryless | Short | 50.00% | 50.00% | 50.00% | Fail / Fail |
| Memoryless | Long | 50.00% | 50.00% | 50.00% | Fail / Fail |

For schema41, both probes scored 100% in each of the six seed/regime cells.
Its shuffled state-probe controls ranged from 56.25% to 64.58%; shuffled
read-probe controls ranged from 35.42% to 67.71%. Thus shuffled controls need
not individually equal chance, but the preset margin passed in every cell.
TAC shuffled controls ranged from 50.00% to 51.04%. The memoryless controls
remained exactly at chance and had zero paired state/read/probability gaps.

## Paired historical distinctions attenuate

These are mean L2 distances between opposite-outcome paired recurrent states,
averaged over the three seeds. Scales depend on the architecture and cannot
be treated as a fair cross-architecture information measure.

| Architecture | Training | H32 | H128 | H512 |
|---|---|---:|---:|---:|
| schema41 | Short | 0.70777 | 0.69335 | 0.69335 |
| schema41 | Long | 0.72085 | 0.70911 | 0.70911 |
| tac_joint | Short | 0.07614 | 0.03008 | 0.001531 |
| tac_joint | Long | 0.05317 | 0.02194 | 0.000864 |
| tac_encoder | Short | 0.06413 | 0.02838 | 0.001597 |
| tac_encoder | Long | 0.05811 | 0.02709 | 0.001626 |

Both TAC arms still had nonzero paired state and read-input differences at
H512, yet their original H512 decisions disagreed on **0 of 144 pairs per
arm/regime**. For tac_joint, the mean paired positive-probability gap fell
from 0.003270 at H32 to 0.00004256 at H512 under short training, and from
0.0005389 to 0.000009467 under long training. Prefixes preserve the original
absolute forecast endpoint; they do not replace DEV-018's H512 endpoint.

## Interpretation and next action

The structured arm is a positive control: the probe recipe can recover the
causal labels from some frozen representations. TAC has no comparable
positive recovery result. Consequently DEV-020 does **not** establish the
proposed pattern of recoverable state information blocked specifically by the
reader. It also does **not** prove that the information has been erased.

Attenuation is consistent with the decay/interference hypotheses, but no decay
factor, erase contribution or key alignment was intervened on in DEV-020.
Distances can retain irrelevant information. Moreover, dense-state probes use
6,158 input features but only 288 training episodes, and the normalization's
0.05 standard-deviation floor can leave small signals difficult to learn.
Probe failure remains compatible with sample, conditioning, capacity and
optimization limitations. No fair probe-budget comparison across architectures
is claimed.

The [DEV-021 proposal](DEV-021-SUFFIX-TRANSPORT-PROPOSAL.md) therefore targets
the trained frozen recurrence under matched suffixes before choosing a memory
redesign. It proposes separating multiplicative decay and erase effects with
explicit difference-dynamics interventions on fresh histories. This is a
proposal only; no DEV-021 result or architectural improvement is claimed.
Native TAC reliability, memory-attribution and overall success gates remain
failed. Keep the research PR draft and unmerged.

## Execution correction and validation

[Previous run 38832325450](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38832325450)
completed two checkpoint jobs, failed four, and skipped the audit. Those failures
were a float32 comparison bug: schema41 short seed43 had 91/96 correct,
but float32 mean `0.9479166865348816` differed from archived exact-count ratio
`0.9479166666666666` by about 1.99e-8, exceeding the 1e-9 assertion tolerance.
The fix uses integer counts divided by sample counts, preserving the exact
archived endpoint check. It does not alter checkpoints, probe recipes or
decision thresholds. Earlier batch-shape and tie-handling corrections remain
in the branch history; failed attempts are not scientific outcomes.

Validation: 1,048 local tests passed, focused regression and lint checks passed,
test-count provenance is synchronized, and the corrected full workflow passed.
A duplicate manual dispatch, run 38832919404, was cancelled to avoid redundant
execution; the completed evidence is solely from run 38832908326.
