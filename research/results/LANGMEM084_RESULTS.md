# LANGMEM084 — explicit byte codes help a fixed readout; overwrite exposes a short-versus-long interference tradeoff

**Completed internal mechanism measurement, not independently qualified.**
Three original-checkpoint jobs and strict paired aggregate passed in
[source run40968863964](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40968863964),
[original aggregate artifact12938912385](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40968863964/artifacts/12938912385).
The replay with pinned PNDS operator parity also passed all four jobs in
[40969058652](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40969058652).
Original source **6ffc14c26728deed653b31e224fdf74d3070996a**;
source/parity bridge **ad6f2f4511be4ab38c8da6e135ac0340f6cdf701**.
[TAC draft#627](https://github.com/FPC-effortless/TAC-OSM/pull/627),
[PNDS partial-math draft#81](https://github.com/FPC-effortless/pnds-lean/pull/81).
Original source artifacts locally reaggregated with exact JSON value equality.

## Frozen source, component budget and complete factorial

Three original078 source-trained499627-parameter checkpoints, original
optimizer/RNG/source-order SHA receipts from40890642312, no original weight
changes. Original079 producer carries4160 FP32 tensor bytes per owner.
PSF/0BSD CPython3.13.0 exact original source commit
60403a5409ff2c3f3b07dd2ca91a7a3e096839c7;083 SHA/length/Git blob verification
for `Doc/howto/regex.rst` and `Doc/howto/argparse.rst`. These documents are
**previously exposed by083**, not newly unseen for084. No new FIT, decoder
training, hyperparameter selection, fitted transforms or independent source.

Observed source bytes0..639; queries128..636 step4; old anchor a=t-age>=4.
956 query-age cells per seed, consisting of **314 distinct owner/anchor byte
positions** reused at different delays. Age denominators256/256/254/190.
Eight arms produce7648 repeated measurements per seed, not7648 independent
outcomes. Three seeds share the source cells; deterministic byte/hash controls
share the same fixed encodings, so repeated seeds are not independent learned
replications of those encodings. Report all arms, all seeds and all strata.

Values: original learned byte+position native value, or fixed signed8-bit
identity c(byte)[i]=(2bit_i-1)/sqrt(8). Keys: original contextual key, or
normalized SHA256(previous4 observed bytes), with no event clock/target in
address. Update: original source rho M(I-beta kk^T)+epsilon vk^T, or
unit residual overwrite rho M+(v-rho Mk)k^T. This is an established delta-rule
control. Native TAC already has separate key/value projections; this trial
does not invent that separation. Each bank8×8 FP32=256 tensor bytes.

Read M_t k_a uses **an external evaluator-supplied old key and clock**.
Decode by fixed cosine nearest to256 original native value codes at clock a
or the explicit byte codebook. No fitted decoder or probability calibration.
Both registered wrong controls use other historical four-byte contexts, same
owner and query clock (see target-overlap limitation audited below): nearest/hard wrong key, and SHA-uniform wrong key. All predictions
freeze before raw[a] is used by scorer. raw[a] legally entered memory only
when historically observed; it is not fed directly into a query decoder.

## All eight registered arms: historical-byte accuracy

Arm name is `value_key_update`. All percentages are seed means, with identical
query-age denominators per arm.

| Arm | Age8 | Age32 | Age128 | Age256 |
|---|---:|---:|---:|---:|
| bytebits_context_hash_overwrite | 10.55% | 1.95% | 0.79% | 0.00% |
| bytebits_context_hash_source | 7.81% | 7.29% | 10.24% | 0.00% |
| bytebits_native_overwrite | 9.51% | 3.91% | 1.71% | 3.16% |
| bytebits_native_source | 9.24% | 7.81% | 11.55% | 7.19% |
| native_context_hash_overwrite | 7.16% | 2.60% | 2.23% | 1.23% |
| native_context_hash_source | 5.60% | 5.86% | 6.56% | 1.23% |
| native_native_overwrite | 6.77% | 3.78% | 1.31% | 2.63% |
| native_native_source | 4.82% | 3.65% | 4.86% | 1.93% |

The native/native/source bank exactly matched actual original slow state at
every source step (max error0 in all three original jobs). Thus the first
row for that arm is an actual original compressed-state inverse, not a
substitute initial matrix or random pretrained model.

Holding native key/source update fixed, explicit byte codes raise recovery
from4.82→9.24%,3.65→7.81%,4.86→11.55%,1.93→7.19% across ages8/32/128/256.
However the same explicit-code arm with **hard wrong addresses** scores
9.11%,7.42%,12.60%,7.19%. The long-age increase does not by itself establish
specific old-event recovery. Against uniform wrong keys it does have a
coarser address-dependent difference; both controls must be preserved.

The value factor also changes per-event value norms: explicit codes have
unit norm; original tanh value norms vary by byte/time/seed. The benefit is
therefore a result for this **representation plus normalization**, not a
causal attribution exclusively to identity separation. No unregistered
norm-matched follow-up was run or silently substituted. Likewise unit
residual overwrite changes both erase/write strength and computation versus
source update. Factorial comparisons describe these fixed interventions,
not an optimized equal-FLOP algorithm contest.

## No standalone accuracy win over the existing FIT-only prior

A read-only comparison with original083A artifact12938940382 from
[run40967076169](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40967076169)
confirms the **same historical target SHA**
`6ada58bdd0d3b715be935cac5fe491004e664ac196ec84e0a5985959f9409884`.
Its frozen FIT-only age-modal byte baseline scores12.50%,13.28%,12.20%,12.63%
on the registered age8/32/128/256 cohorts. Every084 arm is below that baseline
at every age: even the maximum reported correct-address score among all8 arms
is10.55%,7.81%,11.55%,7.19%. This maximum is descriptive, not tuned model
selection. Thus a small correct-versus-decoy channel difference does **not**
translate to a useful standalone predictor here. No new prior fitting, source
read or DEV/model evaluation was performed. The cohort-bound check is saved
in `LANGMEM084_EXISTING_PRIOR_CHECK.json`; it compares the original full
cohorts, not a differently filtered post-hoc subset.

## A narrow short-age signal, not long-history retrieval

Explicit byte codes plus context-hash keys and unit overwrite score:

| Age | Correct address | Hard wrong address | Uniform wrong address |
|---|---:|---:|---:|
|8|10.55%|7.81%|2.34%|
|32|1.95%|1.56%|1.56%|
|128|0.79%|1.18%|2.36%|
|256|0.00%|0.53%|2.63%|

At age8, on the **unique-context** subset, correct9.31% versus hard wrong6.37%
and uniform wrong1.96% (204 source cells). These correspond to19 versus13
versus4 correct cells per seed for this deterministic encoding. By owner:
argparse10 versus8 hard-wrong on99 cells; regex9 versus5 on105. This is a
small descriptive signal, not an independent statistical qualification.
At age128 unique-context accuracy is1.08% versus hard wrong1.08% and uniform
wrong2.70%; at256 it is0%. Aggressive overwrite makes the latest write exactly
readable at its unit key, but that property does not protect old values from
subsequent writes. The source-key/native-value overwrite control has the same
short-versus-long weakness. Neither longer retention alone nor more aggressive
writing alone is established as a general solution.

## Post-hoc audit: wrong contexts can themselves contain the historical target

The registered wrong-address selector allowed any different context at4<=b<t.
A context raw[b-4:b] **contains the requested source index a** when
`a < b <= a+4`. This can supply answer-bearing information through the decoy
key. All of these bytes are observed by query time, so it is **not future-byte
leakage**, but it weakens the interpretation of this negative control. Merely
having no explicit label argument does not establish label exclusion at the
source producer. Correct old key raw[a-4:a] excludes raw[a]; a decoy can differ.
The same source-level risk exists in the inherited083 control construction;
we do not assert a count or effect on083 without auditing its original outputs.
083's correct-key versus address-only comparison remains a separate control.

This stronger historical-target exclusion was not implemented in the registered
084 decoy control. **Do not mark it PASS** because the separate future-poison
tests passed. The original predictions, original controls and all excluded rows
are preserved. `LANGMEM084_CONTROL_AUDIT.json` adds a clearly labeled
**post-hoc, read-only** analysis with zero new model evaluations, fits or DEV
reads. It discards a row if **either** decoy context contains index a and scores
correct/hard/uniform on the exact same remaining rows per seed. Selection uses
only source indices, not target value or prediction correctness. These are not
new decoys, prospectively confirmed controls or fresh unseen validation.

Native hard-decoy overlaps occur14/29/24 times per956 source cells for the
three seeds; uniform11/956 each. Hash hard-decoy overlaps8/956; uniform11/956.
On the retained paired subset:

| Arm and age | Correct | Hard wrong | Uniform wrong | Retained cells per seed |
|---|---:|---:|---:|---|
|native/native/source,8|3.97%|3.56%|1.37%|245 /244 /241|
|bytebits/native/source,128|11.72%|12.80%|2.97%|250 /244 /248|
|bytebits/hash/overwrite,8|10.84%|7.63%|2.41%|249 /249 /249|
|bytebits/hash/overwrite,128|0.80%|1.20%|2.41%|249 /249 /249|
|bytebits/hash/overwrite,256|0%|0.53%|2.66%|188 /188 /188|

The descriptive short-delay separation and long-delay failure remain. The
hard-control long-age result is still not convincing specific recovery. All8
arms and per-seed denominators are retained in the audit JSON. Controls still
differ in their writer ages: this audit removes direct target-position overlap,
not every possible source/context correlation or temporal matching limitation.

## Distinguishing repeated context from erasure

Among the956 source cells, unique-context denominators by age are204/203/185/135;
repeated contexts with conflicting observed continuation bytes are25/26/40/35.
Counts include all observed writes through current t. All arms use the same
context categories and valid denominators. A four-byte address cannot encode
which of several same-context historical occurrences is requested; that
limitation must not be relabeled as pure forgetting.

For bytebits/native/source at age128, unique-context correct4.68% versus hard
wrong5.41% and uniform wrong1.98%, while its all-row score is11.55%. The larger
all-row score is not evidence of a robust unique-event channel. The full
unique and conflicting strata for every arm remain in the aggregate.

Original native keys have uncentered participation rank roughly4.07–4.51
in8 dimensions across six seed/owner cases, versus context-hash7.58–7.83.
Native mean-key norm0.154–0.284 versus hash0.059–0.072. Better geometric spread
alone does not solve long-age recovery: bytebits/hash/source scores0% at256,
and hash overwrite also0%. SHA keys are fixed controls, not learned semantics.

## Raw storage and honest total memory accounting

The256B raw ring includes16B cursor/count metadata and240 bytes of payload.
It recovers retained historical bytes at ages8/32/128 exactly; all age256
queries correctly return unavailable. Byte storage is not an autonomous
context/fact retriever or next-byte language prediction result.

**256B is only the mechanism bank, not full deployed TAC memory.** The runner
also evaluates a shared original4160B producer, all8 diagnostic banks2048B,
and evaluator-only key/context/history indexes. It caches640 native codebooks
of256×8 FP32 values (5,242,880 evaluator bytes), plus the fixed byte codebook
8192B; these can be recomputed from frozen weights or bit rules but are NOT
hidden in the256B budget. They are not new learned parameters. Owner/clock
metadata and Python allocator overhead are outside tensor payload counts;
ring cursors are explicitly inside its256B. A proposed integrated replacement
would need a separate actual total-state/latency audit. No same-full-budget
architecture or edge-deployment claim is made. Update timings per arm/seed
are saved; work differs between source and overwrite modes.

## Executed checks, preserved failure and remaining obligations

All original checkpoint/model/source hashes unchanged, native bank parity
exact and original128-byte full graph logits within5.8e-6. Per seed two-owner
actual source future-poison pairs, including all8 bank updates, source factors,
logits/state and hash producer, pass bitwise; current-byte changes leave the
historical writer key unchanged, explicit byte values differ; source shams
pass bitwise.23 constructed engineering/probe tests pass. Pinned independent
PNDS Python operator comparison passes for256 byte codes and64 randomized
Float64 operator pairs, max error8.89e-16 at1e-12 tolerance. No global pipeline
or source IEEE/AD proof is inferred from these scoped checks.

Initial PNDS Lean runs40968873496 and40969055889 failed because signed-bit injectivity left
the arithmetic contradictions -1=1 and1=-1 unfinished; this failure is
preserved. The repair changes proof tactics only. Lean kernel build, Python
reference tests, historical reciprocal-index checks and explicit axiom audit
passed in [40969135827](https://github.com/FPC-effortless/pnds-lean/actions/runs/40969135827)
at proof **4f24b054b413efb2564892215401398c5c339e6f**. No `sorryAx` counts as a
proved theorem. Current bridge head also passed in
[40969555828](https://github.com/FPC-effortless/pnds-lean/actions/runs/40969555828).
Read-only structural audit/reaggregation CI passed in
[40970032497](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40970032497),
with no neural code, source document read or new predictions. A later unchanged
source/bridge replay also passed in
[40969558910](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40969558910).
Formulations cover exact-real overwrite/read, unit-key latest
write, orthogonal-read preservation and unscaled signed-bit injectivity only.
SHA/Float32, full model AD, autonomous address selection, integrated deployment,
independent-domain evidence and coordinated canonical registry merge remain OPEN.

No new model fits, optimizer updates, GPU quota, protected TEST or merges.

The useful direction is **explicitly checkable value identity with selective,
interference-aware writes and meaningful retrieval objectives**, not a claim
that this fixed byte/hash control is already an improved TAC. Any trained
implementation needs a separately frozen100k/matched-budget protocol and
fresh independent development source. These results authorize no such fit.
