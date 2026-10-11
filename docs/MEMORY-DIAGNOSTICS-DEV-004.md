# Memory diagnostics DEV-004: fixed development protocol

The DEV-003 model scored 54.86% on historical answers, 70.83% on short
post-commit carry and falsely accepted 13.89% of negative-evidence writes.
This successor investigates these failures without changing its checkpoints.

The machine-readable protocol is committed before diagnostic measurement.
This is a fixed **development** design, not confirmatory preregistration.
The design lives under `development_protocols/` because it deliberately does
not use the repository's confirmatory experiment-contract schema. Moving that
unchanged document out of `contracts/` preserves its contents and freeze history.

## Questions

1. Does the relevant bit still exist in the final recurrent state? Measure
   exact counterfactual state equality and read differences versus target age.
2. Is useful content lost during writes or inaccessible to the read/compute
   path? Compare native outputs with a separate trained frozen-read probe,
   uniform expert reading, compute bypass and explicit oracle diagnostics.
3. Does the post-commit benefit survive 1/4/16/32 genuine new competing writes?
4. Can a threshold calibrated on separate cases reduce verifier false accepts,
   and how much does it increase false rejection?

## Interpretation boundaries

The read probe sees only frozen expert read vectors and learns the past bit
from supervision. It does not receive the target index, address or current bit.
The answer transformation for this diagnostic is evaluator-side XOR, so probe
performance is not full learned model performance. Failed decoding could reflect
probe optimization/capacity, distribution shift or erased information; it does
not prove that information is absent. Exact paired-state equality, however,
proves indistinguishability for a deterministic state-only decoder on that pair.

Target-only, target-last and exact-address arms use evaluator privileges. They
are labeled oracle interventions and cannot support runtime capability claims.
Uniform read and compute bypass change the distribution seen by the trained
answer head; poor results alone do not establish causal module necessity.

Post-commit partners have identical pre-write memory and present inputs, then
opposite past write values. New intervening writes have identical addresses and
payloads in both partners. No-commit/reset controls have an exact 50% ceiling.
Swapping the pair's memories makes accuracy complementary by construction;
this is an intervention check, not independent statistical confirmation.

Verifier calibration uses an independent deterministic namespace, a fixed
threshold grid and asymmetric error objective. Evaluation covers smaller
evidence margins as well as the original positive/negative regimes. In this
synthetic task, a nonlearned evidence-sign rule is an appropriate perfect
reference. Calibrating a threshold does not improve the neural representation
or create a general reasoning verifier.

Every seed/condition is reported. Descriptive seed-level intervals are based
on three independent model/probe seed pairs and remain exploratory. Neither
probe training nor calibration changes the frozen parent models.
