# Research reconciliation — 20 October 2026

**Goal remains ACTIVE / NOT COMMERCIAL-QUALIFIED.** This review checks all **58
TAC-OSM pull requests #222–#281** (236 and240 are not PRs), including the48
additional studies/policy PRs #230–#279 missing from the maintenance working
ledger. It also checks the PNDS Lean companion and component-repository recency.
No original result, model weight, source freeze, holdout gate or PR is rewritten
or merged. See [inventory.json](inventory.json) for every pinned head, changed
file, document SHA256, current check and explicit review scope.

## What this audit actually verifies

PR bodies, source-frozen result/protocol documents, changed-file lists and actual
GitHub check/run metadata were inspected; selected KuaiRec model/feature/parser
code was read. **58 cited execution records** were queried:49success,7failure,
2cancelled. These include preserved failures, not58independent successful trials.
This review downloads no original checkpoint archives and does not independently
reproduce every published number. Prior maintenance checkpoint audits remain
separate evidence. No new holdout outcomes were opened for the review.

At the snapshot:22PR heads have only successful checks,7have successes plus
skips,23only skips,5no checks and1a failure. **PR230's current smoke fails**;
its older successful preflight remains in the report. Cause of the latest failure
was not established here. Skipped guarded first-exposure jobs protect opened
cohorts; they are neither a fresh replay pass nor a numerical model failure.
The run index preserves actual statuses rather than calling all CI green.

## Important cross-study correction: MetroPT holdouts already exposed

The source SHA256 is identical in REAL-MAINT-001 and the later ME compressor
studies: `aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a`.
REAL001 frozen source `00037395c12b77c9e32ac38278793e37bc9882d2` parsed the
entire source and evaluated3131test targets, March1 09:00–August31 20:00 2020.
The original test tensor, provenance, first-seed score/report hashes were checked
against its stored manifest, not newly generated. Source code constructs test
from March1 onward and explicitly evaluates it after fitting.

| Later claimed untouched week | ME result / PR | Earlier REAL001 test targets within it |
|---|---|---:|
| March2–9 | ME004F /249 |124|
| March9–16 | ME006F /252 |111|
| March16–23 | ME007F /254 |137|
| March23–30 | ME008F /256 |138|
| March30–April6 | ME009F /258 |117|
| April6–13 | ME010F /260 |119|

These later studies retain valid **recorded frozen-model scores on real data**,
but their weeks are **not program-wide untouched holdouts**. Aggregation/windows
and forecasts differ, and this overlap does not prove that later training read
future targets or invalidate the score arithmetic. It does remove the claim of
new independent first exposure across the whole program. Preserve the original
reports and attach this correction rather than silently rewriting history.
[Machine-readable checked overlap](metropt-global-exposure-overlap.json).

**REAL002's MetroPT2 2022 recording is different. Its July B stays unopened.**
Do not conflate that reserved period with the exposed 2020 MetroPT3 recording.

## Scientific synthesis and supersession

- **Entity isolation is useful; a unique TAC memory advantage is unestablished.**
  SEQ005's update failures were followed by SEQ006C's20-seed success atL512/E6;
  a simpler entity-routed GRU matches every cell. SEQ007/008 remove some teacher/
  event-position shortcuts on validation, but have only2seeds. SYS1–SYS2 probes
  expose additional seed sensitivity and do not demonstrate deliberative search.
  Do not keep listing late-overwrite training as entirely untested, or call a
  routed-GRU win over global GRU a TAC-specific discovery.
- **CASM's named dependency tests were attempted and closed.** DEV064 lacks
  registered practical headroom; DEV064B obtains headroom but all architectures
  nearly tie and closure is only~37.4%, including its cost oracle. The fallback
  generator defect remains preserved. The old goal's "DEV064 pending/unexecuted"
  status is stale. No CASM superiority survived these tests. Any new real-action
  hypothesis needs its own justification, not a renamed synthetic benchmark hunt.
- **Full integration is not contribution evidence.** ME005 combines native
  recurrence, assemblies/graph, surrogate CDL-PHR, bounded association, neural
  operators, attractor refinement, rhythmic gates and evidence lifecycle. Its
  full adapter barely improves on a residual-only head and is slower; ablations
  are small or inconsistent. ME006's direct future-supervised adapter outperforms
  its slower Hebbian association. Preserve the implementation, not an unsupported
  claim that every integrated module earns deployment cost. Original CDL LM
  teacher-distillation and symbolic CASM discovery remain distinct open claims.
- **Native real forecasting is useful in places, unstable across shifts.**
  ME004's favorable compressor averages were weakened by a better-trained GRU
  in ME007F; every learned arm loses to persistence in ME008F. The ME009 gate
  and ME010 event mixture fail their incremental gates. These weeks now carry
  the exposure correction above. ETT quantile/conformal models fail strong
  persistence and cross-site calibration; mathematical interval ordering does
  not guarantee coverage. Air-quality models beat persistence on two transfer
  cohorts, but GRU wins on mean and anonymous routing does not help.
- **KuaiRec identifies feature/readout/optimization limits before a new memory.**
  ME015/016 native32 has modest favorable3-seed comparisons on some real users;
  the proposed extra identity path fails. ME019/019F native128 does not reliably
  improve and loses to strong GRU128. ME020 shows contraction without collapsed
  gates and with decodable old information; random-state control and better GRU
  despite smaller gradients block a simple causal forgetting explanation.
  ME021's +.001570 paired user-clustered interaction survives calibration, but
  the225parameter MLP's .169752 Brier versus best recurrent .200854 is much more
  consequential. The latter uses engineered causal summaries and known metadata;
  not an equal-resource architecture win, and not a maintenance policy result.
- **Engineering/proofs have narrower positive results.** ME018 preserves exact
  native equations/gradients while reducing measured CPU runtime. Keep it as an
  optional execution optimization. PNDS Lean PR4 builds successfully and proves
  explicit admission/convexity assumptions; it does not authenticate real
  receipts, prove predictive correctness or certify the whole agent.
- **Commercial evidence remains incomplete.** REAL003 is a qualified measured
  room-temperature point-forecast component with its Fahrenheit correction;
  not native-TAC or repair-policy qualification. REAL004 actually fitted Malawi
  repair-flow outcomes but lost to its simple reference. SCANIA supplies real
  telemetry and recorded repair/censoring, not repair efficacy or actual prices.
  No current experiment proves maintenance ROI, independently verified action
  benefit or deployment safety.

## Change to current execution

The working ledger missed PR237's standing real-world evidence and parity rules.
They must govern new work, even though that policy PR is draft/unmerged. Its exact
pinned bytes are proposed here in [REAL-WORLD-EVIDENCE-RULES.md](../REAL-WORLD-EVIDENCE-RULES.md),
with [provenance](POLICY-SOURCE.md). No new standalone synthetic-accuracy research
may be continued as a capability study; historical diagnostic archives remain.

**DEV069 was stopped administratively, not scored as a failed hypothesis or a
successful model.** Frozen source71a7bb693b9cb4988cb1cdf570ac2a3242e88dfd;
12pilot heads and12fresh heads completed;2additional6-head groups were partially
trained and retain atomic raw/EMA/optimizer/RNG snapshots. The registered20-seed
matrix is incomplete and no test opened. Stop is motivated by discovered policy,
not selection of successful seeds; it cannot support a reliability comparison.
Complete-checkpoint replay checks24EMA+24raw states,1536recorded trajectories,
5655native actions,50standalone decisions and zero local replay drift. A separate
partial-snapshot audit checks24frozen-body states. This is preservation of work,
not renewed synthetic scientific promotion. Fitting source and weights unchanged.

**ME022 remains a proposal, not an executed or promoted experiment.** Its draft
should be reviewed against this reconciliation before source freeze or fitting.
No new memory/SSM/attention conversion is justified by ME020 gradients. The next
real-data registration should first disentangle engineered feature access,
normalization, readout learning and optimizer budget, with dense learning curves,
strong tiny references and matched native/GRU controls. Explicitly separate
compute-constrained and exposure-matched lanes if both budgets cannot match;
never imply full parity from equal parameter counts or wall time alone. No
retuning opened KuaiRec user4/5 as a fresh test or recycling exposed MetroPT weeks.

Maintain the distinct maintenance risk/action-outcome lane. A real outcome source
must still supply causal time order, real action/inspection information, actual
prices and independent physical follow-up before commercial qualification. Public
source discovery remains authorized, but joining unrelated costs or equating a
work-order closure with physical repair does not satisfy this goal.

## Companion scope

PNDS Lean PR4 at8e14f2ececf3a7cd617e82d7e6b18f73703cae8c has a successful build;
its typed receipt assumptions and limits are recorded separately. TAC-transformer
last pushed October17 and CDL-attention's latest research PRs dateSeptember;
no newer October18–20 research PR was found there in this recency check. This is
not a fresh numerical audit of every historical component experiment. The binding
component provenance exclusions, especially gold-leaking REAL017 and CASM heldout
failure, remain in force. No other repository was modified.

See [study-ledger.md](study-ledger.md) for all58PRs and dispositions; original
run metadata and hashes are retained alongside this report. This reconciliation
updates decisions, not scientific results. The model-development goal stays open.
