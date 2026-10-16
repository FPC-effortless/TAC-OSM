# DEV-024 frozen protocol: structured writer learning attribution

Adaptive development study, registered before opening new outcomes. No overall
native TAC or CASM success can be licensed by this study. Contract:
`contracts/DEV-024.json`; preceding proposal remains historical.

## Question and matched interventions

Keep DEV-023 masked-local computation: entity-indexed write/read masks and aging
only on the public entity slice. Four arms use byte-identical initial tensors:
joint; random frozen writer; random frozen event encoder; both random frozen.
The query encoder, read key and prediction head learn throughout. Frozen modules
remain in the differentiable computation, so a frozen writer can propagate
input gradients to a learned encoder. Add learned and random frozen-recurrence
schema41 references. No trained checkpoint initializes any arm.

Total TAC parameters 8,674; active parameters joint 8,674, frozen writer 5,378,
frozen encoder 7,734, both frozen 4,438. Schema references have 4,684 and 4,354
active parameters. Four TAC arms use 768 memory scalars; references use 41.

Five seeds 43/59/71/89/107, initialization/evaluation numerical offset 24,000,
training offset 6,000. These namespaces are fresh relative to DEV-023. Retain
DEV-018 AdamW recipe: 216 updates, learning rate .002, weight decay .0001, clip 1;
short H128 batch8, long H512 batch2. Both expose 1,024 pre-sensing events/update;
short sees four times the episodes, labels and inspection receipts. Timing and
active parameter counts are reported, not treated as matched compute.

## Outcomes and decisions

Use unchanged public-history generator, receipt rule, controller and costs.
Evaluate H32/H128/H512/H1024, delays32/128/512, four roles, three deadlines,
four repetitions and both pair members: 1,152 rows/cell. Primary96 are H512
pending/reissued cancellation at/after due, before sensing. Physical evaluation
covers all288 H512 rows, including selected, model-only and budget-matched
controls. Retain reset, paired-state swap and H1024 diagnostics.

Primary writer contrast: joint minus frozen writer. Required second attribution
contrast: frozen encoder minus both frozen. Each must gain at least five
percentage points with paired five-seed exploratory 95% t(df4) lower bound >0.
Joint must also achieve >=80% primary accuracy and >=10pp swap drop in every
seed. Evaluate each regime separately; a both-regime claim needs both to pass.
Failure of either contrast leaves writer-learning value unproven, and does not
establish equivalence or prove learning never helps. Registered secondary
encoder effects and writer-by-encoder interaction cannot rescue attribution.
Intervals are exploratory; secondary tests have no multiplicity adjustment.

Keep physical gates: every seed failure <=.05, unnecessary service <=.10,
cost/entity <=.80, and mean selected cost <= mean model-only cost with no tuned
tolerance. Report prediction, selected physical and model-only gates separately.
Learning attribution alone does not establish economic utility.

## Evidence and bounded autonomous operation

Freeze code, contract and protocol before training. Match histories, receipts and
initial component hashes across arms; prove unchanged frozen component hashes
and absent gradients, and changed active components with gradients. Save every
checkpoint, raw row, source hash and exact separate-process replay certificate.
The independent standard-library auditor regenerates cases, recomputes metrics,
physical outcomes and paired contrasts, and requires the complete 60-cell grid.

Use the operating loop in `research/autoresearch/program.md`: run this registered
study, audit it, log outcomes, then formulate the next justified question. The
initial batch has an approximately 30-minute compute budget; do not silently
launch additional matrices or tune thresholds after seeing outcomes. Failed
scientific controls stay in the evidence. A partial run has no final result.

Scope remains synthetic fixed-eight-entity public-slot memory. Writer/read mask
effects stay bundled. No entity-count transfer, native TAC recovery, architecture
superiority or full CASM conclusion follows from a successful structured arm.
