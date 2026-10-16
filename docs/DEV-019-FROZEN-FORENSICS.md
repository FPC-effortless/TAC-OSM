# DEV-019 — Frozen DEV-017 prediction-collapse forensics

**Status: POST-HOC DIAGNOSTICS ONLY — not new training or confirmatory evidence.**
Parent: draft DEV-018 PR #174, commit 77e056bfc3a4f55ea9d9f726502b7b1d226c1b77.
Keep the frozen DEV-017 and DEV-018 protocols, checkpoints, baseline numbers and failed
gates unchanged. This work is stacked research, draft and unmerged.

## Why this study exists

DEV-017 showed 41-scalar publicly indexed schema at 91.84% versus TAC joint
57.90% mature future accuracy. All TAC arms scored exactly 50% on H512
pending/reissued cancellation. TAC joint paid 38.30% unnecessary service
on H512 and cost 0.9695 per entity selected, near the **1.0 always-service**
cost before inspection. These are established DEV-017 results, not discoveries
of DEV-019.

Exactly 50% on balanced opposite-past pairs can mean constant decisions, inversions,
uninformative confidence or other failure. Short training H32/H128 versus held-out
H512 complicates causal attribution. Before changing memory architecture, diagnose
*what* the archived trained systems actually predicted.

## Archival inputs and invariants

Read only the three frozen DEV-017 full seed reports (43, 59, 71)
from the audited archive produced by GitHub Actions run 38551851130,
artifact 11886684731, named
DEV-017-COMPLETE-AUDITED-EVIDENCE-NOT-CONFIRMATORY.
DO NOT retrain or select a new checkpoint. Run the analysis on precisely the
1728 saved rows for each of the six arms per seed; match exact ordered
public fingerprints across arms and their logged prediction argmax values.
Enforce published mature and H512 critical accuracy and original physical
cost arithmetic, as well as 128 balanced critical episodes / 64 opposite-past
pairs per seed and arm. No hidden world state is read by any model.

Input complete archive SHA-256 (ZIP): 
9fad2ff62c511a6dc829808823a4706615906dd923ab1552395dc4ffd95983b1.
Individual reports are independently hashed at analysis time.

## Diagnostic endpoints (post-hoc; descriptive, not new scientific gates)

1. **Prediction collapse:** predicted-positive rate vs ground-truth base rate,
   confidence, Brier, log loss, and per-arm classification/logit hashes
   for the held-out critical binary task.
2. **Paired causal discrimination:** does a model *change* its predicted class
   across paired public histories with opposite future targets? Also report the
   absolute probability gap and signed ground-truth-aligned gap.
   An exact 50% aggregate does not prove all pairs are constant.
3. **History stratification:** descriptive cancellation accuracy and positive-rate
   summaries for H32, H128, H512. These are not a matched causal memory-interference
   intervention and cannot isolate training-horizon mismatch from decay.
4. **Physical cost audit:** recompute actual service, failure and inspection costs,
   compare selected, random and no-sensing outcomes, and retain the 1.0
   always-service reference.

Report each seed and arm; do not treat correlated episode rows as independent
scientific replications. Pair/seed averages are descriptive only. No multiple
comparison claims, no learned-writer attribution, no new TAC world-model success
and no novelty claim.

## How to interpret

- 50% accuracy + near-zero pair-disagreement + very high class imbalance
  supports output/classification collapse; it **does not**, by itself, reveal
  whether the underlying recurrent state retained a discriminative trace.
- 50% accuracy + substantial pair disagreement but wrong direction suggests
  label mapping/calibration/readout failure rather than constant output.
- Nonzero signed probability gap in the true direction with poor argmax can
  motivate a **training-only** calibration/readout study, not retroactive
  inference-time oracle repair.
- Increasing errors at long H only motivates a controlled retention versus
  length/generalization study; it does **not** prove destructive interference.
- A cost close to 1.0 signals an economically near-always-service policy on this
  benchmark; compare reference controllers at the same physical cost definition.

## Next independent, prospective research: entity interference / decay

Following this **read-only** audit, preregister and implement a separate,
newly trained study with entity counts 2, 4, 8, 16; matched relevant-event
age and distractor update count; H128/H512 training controls; and writer
interventions for per-entity partitioning, decay, erasure, and addressing.
Track cosine collisions, target state sensitivity before/after distractors,
writer gradients, predicted-positive rates and actual decision costs.
Use at least five training seeds and compare both state-matched and
parameter/compute-budgeted groups. Keep the original DEV-018 claims frozen.
Do not claim the four current memory experts literally limit the system to
four entities: each associative expert is a matrix and can encode multiple keys.

## Running the forensic audit

With exactly three unmodified full DEV-017 report.json paths:

\`\`\`bash
python scripts/diagnose_dev017_predictions.py \
  frozen/seed43/report.json frozen/seed59/report.json frozen/seed71/report.json \
  --output artifacts/DEV-019
\`\`\`

Outputs: analysis.json with per-seed/per-arm hashes and diagnostics,
critical-H512.csv, cancellation-by-history.csv.
Github workflow **DEV-019 frozen forensics** retrieves the existing archive
using actions:read and uploads only these new descriptive tables.

No result exists without verified source report hashes and the command above.
