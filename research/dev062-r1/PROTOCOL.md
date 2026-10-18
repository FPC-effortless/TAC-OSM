# DEV062-R1 — conditional belief-first optimization follow-up

Preregistered before fresh-data fitting, motivated by DEV062A validation KL around
.03–.05 despite successful control. Do not alter DEV062A checkpoints, source,
thresholds or outcomes. Run ONLY if completed DEV062A fails a functional,
posterior-quality or physical gate; if it passes them, skip this added fitting.
This is a new study/budget, not an extension of the old cap or a CASM novelty claim.

Hypothesis: more belief-only optimization and greater proper-KL emphasis during
joint training improve posterior fidelity while preserving learned control.
Recipe bundles warmup and KL weight; do not attribute effects to one constituent.
Same four matched architectures, public-only DAgger, EMA.995 and learned action/
STOP decoder as DEV062A; no inference planner or stopping override. Source/noise
classification is shared training assistance, not retention discovery.

Fresh training initialization/collection seeds92801–92820, pilot92800 (two rates
.001/.003, validation-only selection identical to DEV062A). Fresh corpus2048 and
validation512 at6281001/2; untouched test6281003 and secondary6281004.512belief-only
updates then1024joint updates (1536total), batch64, AdamW wd1e-4, clip1. Proper
posterior KL coefficient4, with original action CE +.25 future KL +.1cost Huber;
no class/risk weighting of probability labels. Collect at768/1024/1280, horizons
2/5/10,64worlds/student/round, shared union, half initial/half union batches.
Cost every acquisition, preserve raw/EMA/all curves and unsuccessful seeds.

New fitting cap90CPUmin including preparation/collection/validation, group240CPU
seconds; two workers, one Torch thread each. No extension. Incomplete20seed matrix
forbids test opening or qualification. Disjoint SMOKE_ONLY before fitting. Freeze
source/protocol before the full pilots and seeds; don't tune from held-out results.

Same fixed-history256/coupled two-entity noisy task and256primary rollouts/model
as DEV062A, plus64each secondary noise20%, inspectionprice.4,H512. Report all20
seeds, proper KL/Brier/empirical calibration, completion, physical faults, actions,
paid inspections and costs vs public/privileged/shortcut and matched neural controls.
Exact known-law public filter/planner remains outside deployment.512/1024training
updates are exposure, not proofs of convergence. This remains posterior-supervised
StageA, not discovery of laws or symbolic CASM programs without targets.

Acceptance unchanged:>=18/20seeds at>=95% completion, median fixed KL<=.02 and
expected Brier excess<=.02; zero primary residual faults; competitive paired-seed
cost interval/Holm gates against GRU/MLP; no commercial or cross-hardware claim
without their independent evidence. Export predeclared seed92801, never best seed.
Report joint-recipe qualification on fresh data, not a causal improvement estimate
against DEV062A's different worlds/initializations. Preserve trade-off regressions.
If this follow-up also fails, retain NO_PROMOTION; no automatic repeated fitting.
