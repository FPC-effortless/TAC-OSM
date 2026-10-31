# LANGCONTROL098: prefix supervision at fixed future-target coefficients

Prospective internal research, not independently qualified. Parent: draft TAC
PR #665, source `8f2d9bd016358d887f9200d3cdba9bee9681313b`.
The original LANGACQUIRE097 source, weights, scores and failures remain history.
This study changes the objective, not any model's memory or prediction equations.
The distinct LANGMATH098 new architecture in PR #668 is not used.

## Question and falsifier

Does removing prefix auxiliary supervision improve useful lexical reuse when
future-target, episode and domain coefficients remain exactly fixed? TAC is the
primary family. GRU, selective SSM and Transformer are separately reported
within-family replications, not a compute-matched architecture ranking.

The proposed improvement is unsupported if the registered query-minus-full
utility gains and causal acquired-state gates fail. A lower training loss alone
does not satisfy the hypothesis. A negative outcome does not prove that prefix
supervision is universally necessary or that persistent memory cannot work.

## Exact intervention

For each observed training episode, Y is the future target segment and P is all
valid next-byte labels outside Y. P includes acquisition-internal and query
prefix labels. Let CE(i,t) be the complete, unreduced next-byte cross entropy.

    L(lambda) = sum_d 0.5 * mean_{i in d} [mean_{t in Y_i} CE(i,t)
                                          + lambda * mean_{t in P_i} CE(i,t)]

Domains are lexical and ARC. The sampler gives two episodes of each per batch.
The local receipt labels `full` and `query` mean lambda=1 and lambda=0 here.
They do not mean the old 097 token-normalized objectives. There is no division
by (1+lambda), surviving label count or total coefficient mass.
Every Y-byte coefficient is 0.5/(domain_case_count * Y_length) in both arms.
Every P-byte coefficient is lambda*0.5/(domain_case_count * P_length).
Padding has coefficient zero. Domain tags and masks are loss-only metadata;
models see identical byte inputs, owners and clocks. Both arms compute the same
complete CE graph; only a fixed numeric coefficient changes.

Explicit coefficients are controlled; effective AdamW updates are not forced
equal. Prefix gradients can interact with future gradients and change clipping.
Record separate Y/P losses, preclip norms and clipping frequency. This does not
create an acquisition gradient path absent in 097: both old arms already had it.

## Source and frozen cohorts

Reuse 097 data.py byte-for-byte, SOURCE_PROVENANCE.json and FROZEN_INPUTS.json.
The original salt, actor IDs, schemas and whole-file/task splits are unchanged.
The provenance file names its parent study intentionally; it is inherited source
evidence, not a newly collected or independently blind dataset.

Primary: authentic PSF-2.0 CPython publisher code at commit
`60403a5409ff2c3f3b07dd2ca91a7a3e096839c7`. FIT has128 lexical episodes,
DEV64 (32 positive reuse,16 negative,16 uniform). Positive names are absent
from the current query but occur in the earlier acquisition bytes. FIT/DEV source
files and identifier inventories are audited separately. Historical exposure to
these public files is disclosed; this is exposed development research.

Secondary: original Apache-2.0 ARC-AGI public authored tasks at commit
`399030444e0ab0cc8b4e199870fb20b863846f34`;24 FIT tasks,8 DEV tasks.
Task-ID separation does not establish unseen rule families or compositions.
ARC output-format compliance is reported separately from exact-grid success.
No theory of intelligence is established by this diagnostic.

Regenerate publisher source hashes/Git blobs and verify the frozen manifest
`c44e95dd75524d152a9d84d277554805cba6a00f4db642f48e2139472a2c8d41`
and FIT bytes `510b83352b0bb3b14791009735ff7680e7bb6bf622319c9c0391fb923c7eeeb1`.
FIT jobs receive only FIT.json and MANIFEST.json, never DEV actors or answers.

## Fixed models and learning budget

Four unchanged source graphs: TAC100521, GRU100661, SSM100404,
Transformer100484 active/deployed parameters. All are in95K–105K and
within1% of one another. Each family has five paired seeds1701–1705 and the two
objectives:40 genuine fits,256 AdamW updates each,10240 updates total.
Paired exact zero-update archives, sample order, B4x272 dense geometry, input
exposure, precision, optimizer and final-step selection are identical.
Learning rate0.001, weight decay0.01, betas(0.9,0.999), eps1e-8, clip1,
foreach/fused off, one-thread CPU FP32. No tuning or early stopping.

All40 zero-optimizer profiles must bind exact new source and pass the paired
complete ATen signature/shape/count gate before fit is enabled. Every actual
training step must reproduce its profile; first/warm optimizer ledgers are
checked across paired arms. The prospectively registered marker contract permits
only the two balanced empty-tensor optimizer profiler markers, retains their
counts and rejects all other non-ATen operators.

Ten seed/objective legs each have a40-minute experiment ceiling; unchanged
family caps are TAC35, GRU8, SSM8, Transformer4 minutes. Aggregate fit stage
is at most400 minutes, profile15, evaluation40. Setup/download/upload overhead
is separately recorded. GitHub Actions CPU is the venue; phone_lab.py plans
every stage. No GPU or paid service allocation. Timeouts preserve failures and
partial checkpoints; steps, cases and seeds are never silently reduced/replaced.

Cross-family FLOPs, hardware work and energy are not matched or claimed.
CPU time is measured, not an equality gate. Within-family mathematical operator
parity is a narrower claim than equal hardware instructions or useful efficiency.

## Prediction boundary and checkpoint integrity

Before any DEV prediction seal all40 final model/optimizer/Python/NumPy/Torch
RNG archives plus original initialized checkpoints, source/order/exposure
receipts and all failures. Check live parameter counts before the first update
and after fitting. Require finite, connected gradients and actual query-to-
acquisition-state gradients; no optimizer run is inferred from a metadata flag.
Cold reload and chunk/full FIT replay precede DEV. Every acquire/query worker
is source-pinned; query workers receive only Q and checkpoint-bound carried
state, anonymous ownership and legal clocks. No target, source path or answer
length enters the neural input. Gold stopping and teacher-forced DEV are banned.

Retain future-target poison/source-feature invariance, entity isolation, complete
state/KV interventions and40 exact fresh-process prediction replays. Full
arbitrary-code information-flow proof is OPEN; implemented checks are scoped.
Source/poison producers can inspect judge data but are distinct from actors.

## Endpoints, controls and decisions

Reuse the frozen 097 evaluator and analysis semantics in this new study.
Generate exactly4 lexical or40 ARC bytes; score first-byte NLL, exact-four
completion, ARC formatting and exact grids separately. Primary lexical analysis
is the32 positive cases. Keep negative/uniform cases as diagnostics.
Controls: carry, complete content reset, matched real donor, coordinate corruption,
and format/length-matched ARC demonstration-binding shuffle. Five unsupported
ARC donor cases remain INCONCLUSIVE for every model/seed, never dropped.
Raw-history Witten–Bell and copy-last-demonstration remain unmatched practical
references; log full costs and do not present them as matched neural arms.

Average seeds within each case, then bootstrap four code-file or eight ARC-task
clusters5000 times with seed97001. Never count seed copies or bytes as independent
cases. These few exposed clusters support exploratory internal intervals.

Lexical NLL gates: full-carry minus query-carry and query-reset minus query-carry
each at least0.01nat, paired95% lower bound>0, at least4/5 improving seeds and
75% improving original clusters. Donor minus query-carry also needs95% lower>0.
Exact lexical utility separately needs query-carry gains over full-carry, reset
and donor with positive95% lower bounds. NLL improvement alone is insufficient.
ARC needs query-carry exact-grid gains over full-carry and reset, positive95%
lower bounds,4/5 seeds and75% task consistency; binding-specific attribution
also requires these criteria against the binding shuffle. Missing controls
make the corresponding attribution INCONCLUSIVE. The joint gate requires all
lexical/ARC gates; practical-reference advantage is a separate hurdle.

The direct-evidence output-execution gate from the broader competence program
is not newly trained here. If output execution remains poor, rule-acquisition
interpretation remains unresolved. This study resolves the prior objective
confound; it is not the entire theory program or an unseen-composition test.

## Mathematical scope and prior work

Reuse episodic memory/BPTT and in-context adaptation as established methods:
Santoro etal., ICML2016 https://proceedings.mlr.press/v48/santoro16.html ;
Dai etal., ACL2019 https://aclanthology.org/P19-1285/ ;
vonOswald etal., ICML2023 https://proceedings.mlr.press/v202/von-oswald23a.html .
Fixed coefficient algebra also appears in draft LANGMATH098; no priority or
new-learning-principle claim is made here. A scoped PNDS Lean/Python companion
will bind this exact objective/source before training. Inherited model proofs
are partial; whole-model IEEE/PyTorch AD equivalence remains OPEN.

Protected repository TEST, certification, safety claims, checkpoint promotion,
automatic merge and architecture superiority are outside this internal study.
