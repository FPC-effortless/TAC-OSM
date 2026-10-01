# TACOSM-C5-EXHAUSTIVE-REFERENCE-CALIBRATION-001

Status: PRE-REGISTERED.

## Problem

The C5 robustness measurement defines capability retention relative to exhaustive
target retrieval. On the current teacher/task construction, exhaustive target
recall falls as the persistent-state population grows:

- M=128: 0.386
- M=256: 0.166
- M=512: 0.053

The teacher is trained on only 48 non-target codebook states while the evaluated
population adds M-dependent decoys. At large M the exhaustive comparator can
therefore fail because the learned representation is being extrapolated against
an expanding distractor population.

## Question

Does matching the teacher's non-target training population to the M-dependent
evaluation population stabilize exhaustive reference capability?

## Registered arms

### current_48

The historical teacher regime: 48 fixed non-target training codes (CODEBOOK[16:64]);
all 16 target codes remain held out; same teacher architecture/objective and
512 training epochs.

### matched_M_minus_16

For each seed and M, train the same teacher on the exact non-target values that
appear in the corresponding evaluated state population:

- 48 fixed non-target codebook states;
- plus the M-64 decoy values used by that M-level task population;
- total training population = M-16;
- all 16 target codebook states are excluded.

This is a calibration intervention, not access to target labels or target values.

## Registered protocol

- H=256
- M={128,256,512}
- K=32 (retained for protocol compatibility; exhaustive reference does not shortlist)
- seeds=10-19
- 100 evaluation tasks per seed/M/arm
- width-16 cosine teacher
- same training objective, learning rate, margin, and negative count
- target approximately 24,576 optimizer updates in both arms; the matched arm uses rounded integer epochs and records actual updates within 1% of the target
- exhaustive cosine retrieval across every one of the M persistent states

## Primary endpoint

exhaustive_target_recall.

For every seed and M, report both arms separately. Aggregate pooled recall and
seed-bootstrap uncertainty; do not substitute task-level bootstrap as the primary
uncertainty estimate because the independent units are the ten seeds.

## Materiality rule

At M=512, the paired seed-bootstrap 95% interval for
matched_M_minus_16 minus current_48 is compared with a fixed +0.10 practical
materiality threshold. A lower bound strictly above +0.10 is material evidence
that matched teacher-population exposure improves the exhaustive reference at
the highest tested population.

The threshold is fixed before measurement and is substantially smaller than the
historical 0.333 absolute decline from M=128 to M=512.

## Pre-dispatch amendment CAL-001-A1

The original registration used 512 epochs for both arms. That would have given the
matched arm substantially more optimizer updates because its training population
grows with M. Before any full measurement, this was amended to an approximately
fixed 24,576-update budget for both arms. The old definition and rationale are
preserved in the machine-readable contract. No result was produced under the
superseded schedule.

No outcome from this study changes the registered C5 frontier verdict.

## Secondary stability analysis

Report delta_512_minus_128 = recall(M=512) - recall(M=128) for each arm with
paired seed-bootstrap uncertainty.

The purpose is to determine whether the M-dependent decline is reduced by matched
non-target exposure.

## Integrity constraints

- The 16 target codes are excluded from both training arms.
- The matched arm may see decoy values actually present in its M-level evaluation
  population; this is deliberate distribution matching, not target leakage.
- No product-key routing or beam selection is used in this study.
- No codebook or task-generation changes are permitted after registration.
- The result remains a calibration result even if exhaustive recall improves
  substantially.
- Actual teacher update counts must be reported; population matching must not
  be credited with an effect attributable only to a larger optimizer budget.

## Reproduction

Smoke:

python scripts/run_c5_exhaustive_reference_calibration_001.py --smoke

Full:

python scripts/run_c5_exhaustive_reference_calibration_001.py

Artifact:

artifacts/TACOSM-C5-EXHAUSTIVE-REFERENCE-CALIBRATION-001.json

Full-run trigger:

[run-c5-exhaustive-reference-calibration-001-full]

## Dispatch record

Registered full measurement trigger applied to this non-design annotation commit. Amendment CAL-001-A1 and the fixed update budget remain unchanged.
