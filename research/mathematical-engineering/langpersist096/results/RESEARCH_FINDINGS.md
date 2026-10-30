# LANGPERSIST096: measured capability limits and next intervention

The completed experiment found no useful lexical-persistence advantage for
the frozen TAC checkpoints and no correct general-pilot answers from any
evaluated model. This is a negative result about these checkpoints under the
registered conditions, not a proof that an architecture cannot learn these
capabilities. No weights changed.

Run: https://github.com/FPC-effortless/TAC-OSM/actions/runs/41121902021

Prospective source: `6361202e6f9ed9e39b6604bbebb9c49cd1de85e8`.
All nine checkpoint replays matched exactly. The run took 326.64 seconds,
with zero optimizer steps and zero GPU minutes. All 82 software checks passed.
The initial confinement failure remains recorded in `ATTEMPT1.json`.

## Lexical reuse

The primary positive cohort has 100 real episodes, evaluated with three seeds.
Retaining TAC state increased first-byte NLL from 4.746952 after reset to
4.754565 nats. The registered reset-minus-carry gain was -0.007613 nats,
with publisher-cluster 95% interval [-0.067019, 0.051793]. Only two of five
publishers and one of three seeds had a positive effect. The retained-versus-
donor interval also crossed zero. Every registered NLL criterion failed.

| Positive-cohort control | Exact four-byte completion |
| --- | ---: |
| TAC retained / reset | 0/300 / 0/300 |
| GRU retained / reset | 0/300 / 0/300 |
| SSM retained / reset | 0/300 / 0/300 |
| Fixed raw-history predictor | 27/100 |

Neural counts repeat the same 100 cases across three trained seeds; the raw
control is deterministic. TAC and the raw control each store 3,904 bytes of
state payload per stream; weights, metadata, temporary work and process memory
are recorded separately. The raw control demonstrates that useful lexical
evidence is available in the observed history. It does not demonstrate learned
structure acquisition. Historical neural training FLOPs are unmatched, so these
results do not establish an architecture winner or deployment advantage.

Post-run timing audit found asymmetric setup scopes: the raw worker's 0.280409
seconds of recorded stages exclude Python/NumPy import and input loading,
whereas neural setup starts before imports. The enclosing raw subprocess took
0.414886 seconds. Preserve the original ledger; use orchestration wall times
for complete process cost and do not treat the reported setup fields as directly
comparable deployment measurements. This does not change the failed quality
gates. A follow-up must align timer boundaries before a cost advantage claim.

TAC still predicts ordinary code: on the separate 100-position uniform cohort,
retained first-byte accuracy averaged 60% and exact-four completion 23.67%.
These diagnostics do not replace either failed positive-cohort gate.

## General capability pilot

For each seed and both state conditions, every TAC, GRU and SSM checkpoint
scored 0/8 on ARC transformations, 0/8 on GSM8K arithmetic and 0/7 on supported
HumanEval programs. The eighth HumanEval reference, task160, uses unsupported
`eval`; it remains inconclusive without replacement.

The 432 saved model-condition-task records comprise 414 judged failures and
18 unsupported-reference records. All 144 ARC outputs lacked a valid grid and
all 144 GSM8K outputs lacked a valid numeric answer. HumanEval produced 83
syntax/indentation failures and 43 assertion failures; most assertion failures
were empty or comments-only function continuations. No judge cases are missing.
Descriptively, TAC's carried state changed 70 of 72 paired generated outputs
relative to reset, while producing zero correct answers. State sensitivity
alone therefore does not establish useful transfer.

Thus the pilot exposes a present instruction/formatting and executable-action
limit. It does not isolate abstract reasoning from the output interface, and
does not demonstrate learned unknown-rule transfer, autonomous verification,
repair or general intelligence. These are eight exposed development cases per
domain, not an independent qualification benchmark.

## Next learning experiment

The first intervention should teach acquisition and later use before adding
memory machinery. The original CODEVALUE003 worker calls `model(z[:, :-1])`
independently on reset 128-byte windows. It never trained the present
128-byte-acquisition plus 64-byte-query task, nor instruction following on the
general pilot. This is a plausible learning mismatch, not an experimentally
isolated cause of the negative result.

Use a new, source-authentic, prospective study with an existing approximately
100K TAC and qualified GRU, SSM and Transformer controls. Train all arms on
matched acquisition/delay/query episodes from FIT-only sources. Carry the
computed state across the acquisition boundary and differentiate the later
query loss through its writes:

\[
 S_A=F_\theta(A),\qquad
 \mathcal L=-\frac1{|Y|}\sum_t
 \log p_\theta(Y_t\mid Q,Y_{<t},S_A).
\]

Mask acquisition targets in the query-focused objective without detaching its
state. Register a full next-byte-loss control on the same chronological input
episodes, so loss placement is distinguishable from longer exposure. Record
both observed bytes and supervised targets, actual forward/backward/optimizer
cost, and equal-token and matched-FLOP regimes separately. Gradient checks must
show that query loss reaches acquisition writes; executing a pipeline is not
evidence that this learning occurred.

Include real lexical reuse and demonstration-to-new-query transformation
tasks in FIT. Reserve source/task-disjoint evaluation episodes before training;
do not fit, tune on, or relabel any LANGPERSIST096 case as new blind evidence.
Retain all failed seeds. Formatting compliance and task correctness must be
separate outcomes. A useful-state claim requires gains with correct acquired
state that disappear under reset, donor and corruption interventions, alongside
the raw-history hurdle and complete acquisition/retrieval/verification costs.

All newly fitted arms must have 95K-105K active trainable and deployed
parameters with pairwise gaps at most 1%, matched real data/exposure/tuning,
the five-seed benchmark, source/proof/parity records and a compute-planner
decision before execution. The historic 500K checkpoints remain descriptive
references. This follow-up design has not been fitted or qualified by this run.

Only after this objective control should a retention or address mechanism be
changed. A bounded-write candidate must couple write size to forgetting,
`w <= c(1-alpha)`, to preserve the proved state envelope. Longer carry alone
does not fix address interference or teach a reusable rule. Learned queries
already exist in LANGMEM086 and must not be presented as an untested novelty.

Machine-readable results and receipts beside this document are copied exactly
from the CI artifact. `ATTEMPT2.json` records hashes, archive location and its
GitHub expiration; retain the full archive for replay beyond that date.

Publishing these reports automatically triggered the same experiment again.
`PUBLICATION_DUPLICATE.json` preserves that post-outcome run separately; it is
not a new prospective trial or additional independent evidence. All generated
bytes matched, and the lexical gates and general scores were unchanged.
Cross-run FP32 log probabilities differed by at most 0.00001526; each run's own
fresh-process replay was exact. A verified CI guard now keeps report/workflow-
only commits from repeating model evaluation while retaining software checks.
