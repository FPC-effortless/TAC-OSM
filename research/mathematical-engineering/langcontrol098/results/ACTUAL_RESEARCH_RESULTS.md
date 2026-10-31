# LANGCONTROL098: valid controlled measurement, capability gates failed

Internally measured FIT/DEV research, not independently qualified.
All forty original ~100K models trained and the frozen evaluation completed.
Removing prefix supervision at fixed future-target coefficients did not establish
the registered lexical reuse, acquired-state or ARC transfer capabilities.

## What was actually isolated

The same four model graphs, five seeds, original initial weights, real observed
episodes/order, dense geometry and 256-update optimizer recipe were used in
both arms. Both compute domain/case/segment means with lexical/ARC weights0.5.
`full` means future loss plus prefix loss; `query` means the same future loss
with prefix coefficient zero. There is no renormalization by surviving mass.
These are new objectives, not the old token-normalized097 objectives.

The contrast estimates adding/removing prefix auxiliary supervision under the
registered AdamW/clipping recipe. It does not hold effective updates fixed or
uniquely isolate a newly introduced acquisition gradient path. Both old097
objectives already propagated gradients through acquisition.

## Primary lexical results

First-target-byte NLL on32 positive reuse cases, averaged over five seeds.
Lower NLL is better. Gain is full-carry NLL minus query-carry NLL, so a positive
gain favors removing prefix supervision. Intervals resample the four original
source-file clusters; they are exploratory, exposed-development intervals.

| Family | Full: prefix+future NLL | Query: future-only NLL | Gain [95% cluster interval] |
| --- | ---: | ---: | --- |
| TAC |3.498850 |3.802886 |−0.304036 [−0.390870,−0.209617] |
| GRU |3.843268 |3.365686 |+0.477582 [0.453690,0.504539] |
| Selective SSM |3.822215 |3.577938 |+0.244276 [0.204223,0.320264] |
| Transformer |3.521994 |3.432326 |+0.089668 [−0.019018,0.234821] |

Every family/objective/seed completed **0/32 positive lexical cases exactly**.
The untrained raw-history reference completed8/32. Negative/uniform cohorts
are preserved separately; their occasional exact continuations do not rescue
the preregistered positive reuse endpoint.

The TAC primary prediction worsened on average when prefix supervision was
removed. GRU and SSM improved this surrogate prediction metric, but neither
established useful carried information. Their query reset-minus-carry effects
were0.001032 nats [−0.000706,0.002837] and−0.000039
[−0.000058,−0.000020], respectively.

For TAC, query reset-minus-carry was−0.045901
[−0.231128,0.063810]; donor-minus-carry was0.002687
[−0.050731,0.041016]. There is no reliable specific acquired-state advantage.
Transformer showed a reset effect0.046410 [0.008558,0.080009], but donor
attribution crossed zero and exact reuse remained zero. A reset effect alone
does not establish identifier-specific retention.

## ARC and decisions

Every family/objective/seed solved **0/8 original ARC tasks**. The untrained
copy-last-demonstration reference solved1/8. TAC, GRU and SSM had no valid
carried grid serializations in either arm. Transformer query produced two
valid serializations among40 seed-task outputs, with zero correct grids;
Transformer full had none. First-byte ARC NLL is a dimension/format diagnostic.

All four families failed lexical NLL-plus-state, exact lexical utility,
unknown-rule exact learning, binding-specific attribution, joint learning and
practical-reference gates. Partial NLL improvements are retained as partial
positives. No architecture winner is qualified: cross-family FLOPs were not
matched, and this is four separately gated within-family objective contrasts.

## Execution and independent recomputation

- Original fit/evaluation source: `50bf7f2818f38c31f7c4d8280c1352c06f33bd4f`.
- Completed run: https://github.com/FPC-effortless/TAC-OSM/actions/runs/41153729689 .
- Four families ×five seeds ×two objectives,256 updates each: **10240 updates**.
- All40 original fitted/initial model, optimizer and RNG archives verified;
  twenty paired initial tensor digests match. All10 objective-leg seals checked.
- Every actual forward/backward and optimizer ledger checked; frozen source,
  authentic FIT, sample schedules,100K counts and fixed coefficients verified.
- All40 fresh-process prediction replays matched. Independently reconstructed
  **11520 main rows and320 binding rows**, references, intervals and gates.
  All200 unsupported donor rows remain explicit and INCONCLUSIVE.
- Maximum independent numerical discrepancy:4.44e−16. Saved scalar objective
  decomposition differs only by FP32 rounding, at most4.77e−7. These checks
  establish the implemented invariants, not universal information isolation.
- Instrumented fit stages sum244.4016 CPU-job minutes; evaluation322.6864s.
  Setup/download/upload overhead is separate. Different physical GitHub CPUs,
  instrumentation and incomplete hardware accounting preclude serving-speed,
  energy or absolute-FLOP claims. Zero GPU use or protected TEST access.
- Full GitHub software suite:1930 passed,6 skipped,1117 subtests passed;
 225 focused study checks and36 independent-auditor mutation checks passed.
 Software fixtures and theorem compilation are separate from capability evidence.

Scoped PNDS #99 checks passed11 Lean theorem/axiom audits,11 NumPy tests and
ten original-source hashes. Proof source is `8c50893fab362ad1b1a1e6783862011959bc497e`;
metadata is `05eccd0456f46e4f6544acdab9ab45787a98f373`. Draft reciprocal records
match semantically. Main/master integration and whole-model IEEE/PyTorch AD
equivalence remain OPEN. Neither research PR is merged or promoted.

## Interpretation and next diagnostic

The earlier coefficient confound is resolved for this prospective contrast.
The result does not support removing prefix supervision to obtain the intended
TAC capability under this data/scale/budget. It does not disprove persistent
memory, a different TAC mechanism or a general theory of intelligence.

The next diagnostic dependency is free-running execution with direct task
information. A separate, explicitly unregistered StageA/B draft specifies known
nonidentity grid execution followed by episode-specific mapping acquisition.
It must establish solvability/identifiability and preserve natural no-evidence,
sham, retention and novelty controls. It has not been trained here. Original
097/098 failures, tasks, thresholds and model bytes are not reinterpreted.

The architecture-agnostic theory/competence note is a working framework and
standard information-bound reasoning, not a validated new theory.
