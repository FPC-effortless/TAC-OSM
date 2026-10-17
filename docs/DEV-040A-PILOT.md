# DEV-040A — Frozen-head loss and probe controls on fresh shared cohorts

**Completed development pilot. See [results and evidence](DEV-040A-RESULTS.md).**
The original pre-execution protocol follows; its machine-readable file remains unchanged.
This bounded pilot is separate from the reviewed, unregistered full DEV-040
convergence/tuning proposal. Parent: DEV-039 PR#194 at
`967c0e40420c76ad67c77d48e96a83b67d4af63c`. Preserve all prior experiments.

DEV-039 already balanced roles, but mature reissued target-wait labels were
only96/13,824 supervised bits. DEV-038's focused readout probe and DEV-039's
all-bit prediction loss therefore solve different training objectives.

Three fresh seed blocks4001/4002/4003 train bounded entity-local MTSK,
masked-local DeltaMemory and learned indexed schema41 for216 B2/H512 updates
on matched streams. The original workspace-only checkpoints are unavailable;
these are fresh trials, not reproductions of the original SGD trajectories.
Constructors preserve existing kernels: build_bounded(seed),
EntityAgingControl(masked_local,seed), MemoryBaseline(schema41,seed).
MTSK/masked have768 FP32 persistent scalars, schema41 has41. Parameters and
compute differ; supplied indexing is a structural prior. CASM/CDL/OSM are not
replaced or validated by this controlled memory interface.

Freeze each backbone and compare five heads on its exact existing44D pre-head
features: untouched original, uniform/refocused original MLP, uniform/refocused
linear head. The two losses within a family start at identical head weights.
MLP44-64-2 copies the trained original and uses lr.002; linear44-2 starts fresh
and uses lr.02, both500 full-batch AdamW updates. These fixed recipes do not
establish optimally tuned head-capacity superiority. Uniform CE remains the
control. Balanced CE allocates half its weight to mature target-wait labels
and half to all remaining labels; mature role counts are equal. No role/target
metadata or hidden simulator fields enter model inference.

New shared split namespaces supply576 head-training,288 calibration and1,152
evaluation worlds, each H512 across three delays, four roles and three deadlines.
Opposite-future pairs remain together. Fresh namespaces/fingerprints establish
within-study separation, not unseen configurations or independent confirmation.
Within each seed all15 heads,18 probes and calibrators are saved before any
evaluation cases are generated. Concurrent jobs use a frozen recipe; this is
not the globally sealed20-seed full study. No evaluation-directed tuning.

Primary: **bounded-MTSK balanced-minus-uniform MLP reissued-cancellation
accuracy**, before sensing, at/after maturity;192 target worlds per seed.
Support requires mean>=5pp, paired three-seed exploratory95% t(df2) lower>0
and no negative seed difference. Otherwise report negative/inconclusive.
Other heads/architectures are prespecified secondary development comparisons.
No best-arm endpoint replacement and no confirmatory Holm claim.

Train logistic probes on the96 mature reissued head-training worlds using
train-only standardization and a fixed500-step recipe. Compare trained features,
fully random initial-model features, trained encoder/query/adapter with writer
reset to its matched initial weights, paired-shuffled training labels, query-only
and current-event-plus-query controls. The first four use44D features, query-only14D,
current-plus-query30D. Reset-writer is a post-training intervention with potential
distribution shift, not matched frozen-writer retraining. A successful random
reservoir probe still demonstrates retained state information; selectivity
addresses a training advantage. Runtime reset/swap tests are also retained.

Each head uses both the physical threshold.25 and a threshold chosen solely
on calibration cost from the fixed grid in pilot-protocol.json. Cost is
4*failures+services+.1*inspections, per eight-entity episode. No evaluation
inspections occur, so no sensing benefit can be claimed. Report absolute
failures, unnecessary services and cost, alongside always-service, never-service
and the exact public-history controller. Cost improvement cannot hide increased
failures. The utility contrast separately checks mean calibrated cost nonincrease.

Evidence must contain every backbone/initial/optimizer state, head/probe fit,
calibration and raw evaluation row. Separate-process replay regenerates exact
features/predictions and refits all heads/probes. A stdlib audit regenerates
worlds/labels, action choices, calibration scores, physical costs and summaries.
Raw seed artifacts upload before aggregate results are published. The complete
archive and compact tables receive retrievable GitHub links and checksums.
Tests/replay establish integrity; they are not independent scientific replication.

Full DEV-040 remains pending capacity/compute preflight, an implemented
configuration split, a complete statistical family plan, durable historical
archives and a separately frozen tuning/reliability budget. No original CASM,
CDL or six-expert MTSK candidate is silently replaced by this pilot.
