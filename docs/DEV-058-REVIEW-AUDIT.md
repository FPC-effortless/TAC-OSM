# DEV-058 review audit: teacher fidelity, admission labels and stale reads

This is a retrospective audit of the frozen DEV-058 checkpoints and 576
recorded test worlds. It does not retrain or change them. The new prospective
DEV-059 comparison uses separate seeds and worlds. Software portability,
physical repair, autonomous completion and architectural superiority remain
separate claims.

## Teacher and public-information reference

A public latest-record controller reconstructs current statuses and disclosed
parent relations from public records only. Its action matches the privileged
safe-progress teacher at **every decision** on these exact-history worlds.
Both complete 576/576 worlds. There is no demonstrated private-information
advantage in DEV-058; this equality is checked, not assumed.

| Frozen controller | Complete /576 | Physically repaired /576 | Actions/world | Total cost/world |
|---|---:|---:|---:|---:|
| Teacher / public reference | 576 | 576 | 3.06076 | 2.26609 |
| Learned seed 7581 | 576 | 576 | 3.18229 | 2.34266 |
| Learned seed 7582 | 560 | 575 | 3.32292 | 2.51830 |

| Cost component/world | Teacher/public | Seed 7581 | Seed 7582 |
|---|---:|---:|---:|
| Actions including terminal fault penalty | 1.23646 | 1.30938 | 1.47986 |
| Public observations | 0.87394 | 0.87516 | 0.87688 |
| Logical admitted writes | 0.15569 | 0.15813 | 0.16156 |

Paired total-cost gaps against the reference are +0.07656 and +0.25220
experimental units/world. Perfect completion did not establish teacher-level
efficiency. Every initial observation is paid. The reference uses the same
signal-write accounting convention; its bookkeeping charge is not measured
physical memory traffic. The safe-progress reference is not a proved global
cost optimum or an industrial ceiling.

The action audit makes the excess cost concrete: the teacher makes zero
repairs on already healthy targets, seed 7581 makes 70, and seed 7582 makes
157. Seed 7582 also makes ten repairs blocked by a still-faulty parent; the
teacher and seed 7581 make none. All three record 86 safe-but-unsuccessful
repair attempts. These categories distinguish stochastic repair failure,
unnecessary servicing and incorrect dependency handling.

## Completion failure is broader than a stop-calibration diagnosis

All sixteen seed-7582 failed completions miss the stop action before the
budget expires. Fifteen end physically repaired; one ends with two faults.
Physical repair is therefore 575/576 (99.826%), while completion is 560/576
(97.222%). Both are required metrics.

The full neural state and action path was replayed to inspect these failures.
Across all **126 decisions made while physically healthy inside the failed
worlds**, the highest-scored status record for the selected repair points to
an **older fault=1 record for the same entity**, while a later public status
record says fault=0. None of those top records is the latest status record.
The mean learned completion probability at those decisions is 0.4924; mean
stop probability is 0.3735. This is not a uniform pattern of a calibrated
completion estimate being ignored by a stop head.

This is correlational evidence of stale retrieval accompanying unnecessary
repairs. A maximum read score is not the whole top-k weighted representation;
no causal claim that this alone explains every failure is made. The behavior
is a capability/reliability problem in state estimation and/or control, not an
established psychological condition such as panic or over-caution.

The policy reads action-conditioned latent states and learned cost/completion
signals. It does not directly read the per-entity outcome-head vector. Outcome
prediction scores therefore measure an auxiliary head's fidelity. Zeroing the
latents removes candidate information wholesale; that lesion does not alone
establish semantically faithful model-based planning. Unseen action-sequence
composition and semantic forecast interventions remain useful additional tests.

Adding four actions to each budget increases completion only from 560 to
562/576, while physical repair remains 575/576. Default seed 7581 remains
576/576. Slack alone does not remove the failure pattern. The longer-budget
probe is post hoc and uses the unchanged model, not a new acceptance rule.

## Admission follows the supplied label base rate

| Frozen seed | Observations | True positives | False positives | False negatives | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| 7581 | 50,409 | 4,554 | 0 | 0 | 100% | 100% |
| 7582 | 50,508 | 4,653 | 0 | 0 | 100% | 100% |

True negatives are 45,855 at both seeds. Signal fractions are 9.034% and
9.212%; skipped logical-write fractions are their complements. This is
successful supervised classification of the explicit public kind labels. It
is not evidence that admission discovered future decision relevance beyond
those labels, or learned when a repeated truthful receipt is unnecessary.

The distinction between logical writes and implementation work is especially
important. `LearnedAgent.observe` computes admission, addresses and event
encoding for every observation. It also constructs the dense memory-update
tensors when a gate closes. A closed gate preserves the represented state;
it does **not** establish that physical tensor writes, memory traffic or total
compute were skipped. The 0.02 logical-write charge is a synthetic optimization
term. No 91% computational saving follows from this experiment. CASM's skipped
operator dispatches are a different mechanism; net speed/memory benefits still
need a measured quality-matched comparison.

## Reliability uncertainty

Pooled descriptive two-sided 95% Wilson intervals, conditional on each fitted
checkpoint, are approximately 99.34–100% for 576/576 completion and
95.54–98.28% for 560/576. The familiar zero-failure rule-of-three produces an
approximately 99.48% lower success bound **one-sided**, under its Bernoulli
assumptions; it is not the same two-sided interval.

These are stratified simulated worlds and fixed trained models. They do not
measure reliability across retraining, new task families, new sensors or new
identities. Two seed outcomes permit a sample variance with one degree of
freedom, but cannot estimate training variance robustly. The prospective
DEV-059 grid uses ten fresh initializations per arm and paired worlds, and
reports those two sources of variation separately.

## What a privileged-teacher experiment must establish

Distillation can be useful without an information gap, for example for model
compression or replacing a deployment planner. Privileged information is not a
necessary condition for its value. Conversely, an oracle with private state is
not automatically an achievable student target. Identical public histories can
correspond to two hidden fault states requiring different privileged actions;
a deterministic public-only student cannot copy both at the same decision.
A regression test constructs exactly that counterexample.

A partial-observation study must include a public-information-matched belief
reference and separate teacher regret into an information gap and a learning
or control gap. Pure imitation of a private-state teacher that never inspects
also need not teach a student when to acquire information. Observation costs,
noisy/missing signals and information-seeking behavior must be part of the
experiment. DEV-058 did not establish these abilities. Neither its clean
PyTorch export nor Veritas replay demonstrates transfer to industrial worlds.

## Reproduction

```sh
python scripts/audit_evidence_gaps_dev059.py \
  --directory results/dev058-main --output results/dev059-dev058-audit.json
```

The DEV-058 source, weights, original archive, CI portability amendment and
failed worlds remain unchanged. The audit preserves the teacher/public
reference trajectories, per-kind admission counts, per-decision failure
telemetry, cost components, intervals and budget-slack trajectories in
DEV-059's evidence archive. Native replay is still based on shared synthetic
physics and is engineering verification, not independent physical replication.
