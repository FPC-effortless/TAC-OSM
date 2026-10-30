# LANGMEM086 — trained autonomous native read path helps next-byte loss; historical-content retrieval remains unverified

**Completed single-family internal research, not independently qualified.**
All three fresh100521-parameter model fits, full checkpoint/optimizer/RNG seals,
bitwise cold replay, first new Rust-document evaluations and strict aggregate
passed in [original run40979123121](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40979123121),
[original aggregate artifact12944401672](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40979123121/artifacts/12944401672).
[TAC draft#629](https://github.com/FPC-effortless/TAC-OSM/pull/629),
[PNDS draft#83](https://github.com/FPC-effortless/pnds-lean/pull/83).
Full source graph implementation8f82adda80c67d3eec0fa778b35af62c6f9160c7;
explicit first-fit protocol/workflow commitad8520427eba761f4bbaa2aff84539c8e31516f9.
No original078500k checkpoint modified. No GPU quota, protected TEST, promotion
or merge. This experiment **does train new models**, unlike084/085; do not
label it no-fit or imply all TAC weights everywhere remained unchanged.

## Primary real-source result

Lower NLL is better; mean over three newly trained seeds on exactly the same
1792 actual next-byte target positions in two new Rust owners.

| Prediction/control | Mean next-byte NLL | Difference from full model |
|---|---:|---:|
|Full learned model, own current query|**2.687875**|—|
|Fast native read zeroed|2.770153|+0.082278|
|Slow native read zeroed|2.688784|+0.000910|
|Both native reads zeroed|2.778352|+0.090478|
|Native value rows permuted at read|2.760618|+0.072743|
|Same sampled-FIT byte-frequency prior|3.252751|+0.564876|
|1928B raw-history4-byte-context count predictor|2.967913|+0.280038|

Full-model next-byte top1 accuracy **26.4695%**, FIT prior **15.2344%**.
The source-controlled ring uses tau8 shrinkage toward that exact FIT prior,
not an independent fitted neural comparator. No integrated ring-augmented
language model, trained affine-memory baseline, GRU or Transformer is compared.
These references establish performance beyond simple frequency/context storage
on this cohort, **not architecture or clipping superiority**.

|Seed|Full NLL|Fast-zero minus full|Slow-zero minus full|Both-zero minus full|
|---|---:|---:|---:|---:|
|1701|2.706611|+0.091007|+0.002087|+0.103571|
|1702|2.679956|+0.072441|+0.000104|+0.075596|
|1703|2.677058|+0.083385|+0.000538|+0.092265|

All three seed-average slow-zero differences are positive, but extremely
small. One document/seed (1702, enum chapter) has a negative slow-zero delta
about−0.000075; no universal slow-lane gain is claimed. Both-zero deltas are
positive for both owners in all three seeds, ranging roughly+0.070 to+0.127.
No significance calculation treats repeated document bytes/seeds as independent
publisher samples. The nonlinear lane effects are not additive at the decoder.

## What this proves and what it does not

The model's inference API receives **only observed bytes and its own state**,
with owner and clock metadata. A learned q_t uses current observed lexical
context and prior learned upper state; no evaluator-supplied old key, source
index, target byte, byte-history table or reward enters forward. This is a
complete end-to-end trained compact TAC, not another tiny readout trained on
an unchanged500k backbone.

The read controls establish dependence on the native read path under those
interventions. They do **not** isolate old event content. Matrices are read
**after the current observed byte is written**; zeroing/permuting that read
also removes/distorts current-byte features. Prior upper/concept/aligned/wave
state is held fixed and can already carry earlier read information. No result
here distinguishes remembered old values from extra current-write features,
context-conditioned statistics or distributed predictive features. Read lesions
are also out-of-distribution. A clean past-matrix/current-write-controlled or
historical replay study is needed before claiming old-content recovery.

The preregistered autonomous-query diagnostic further limits the interpretation:
224 every8-clock query cells/seed, with a matching past observed4-byte context
within256 bytes on25% (56/224). **Closest-key exact-context top1 hits are0/56
in every seed.** This proxy therefore does not demonstrate specific historical
context selection. Learned q and decoder may use sign/coordinate/distributed
representations rather than positive nearest-key event retrieval; the zero
proxy is not a proof that no historical information influences predictions.
External logged old keys are for this diagnostic only and never enter inference.
No direct past-byte reconstruction or semantic/factual retrieval is established.

The current result is thus a useful **autonomous bounded-state next-byte model
with a native-read contribution**, dominated by its fast lane under this short
training recipe. It does not yet validate the hoped-for long-term preservation
or meaningful factual retrieval of the proposed two-timescale architecture.

## Fresh source, controlled FIT and actual training

Original genuine PSF CPython3.13 source commit
60403a5409ff2c3f3b07dd2ca91a7a3e096839c7,13 original tutorial FIT owners plus
registered Unicode FIT supplement=14. Git blob, length and SHA256 verified;
086 loader reads only FIT, excluding old DEV. New contiguous257-byte windows
never cross owners. `FIT_PLAN.json` freezes all original per-seed source-order
and raw-window hashes.160 AdamW updates/seed, batch4,256 genuine next-byte
targets/window=163840 training targets/seed; LR.001, wd.01, globalgradclip1.
No early stopping, auxiliary objective, trial search, calibration or DEV tuning.

|Seed|Actual updates|FIT next-byte targets|First FIT loss|Last FIT loss|FIT CPU seconds|
|---|---:|---:|---:|---:|---:|
|1701|160|163840|5.5482|2.5442|370.40|
|1702|160|163840|5.5962|2.3728|215.11|
|1703|160|163840|5.5890|2.5074|362.58|

First/last losses use different sampled FIT batches and are not a held-out
learning curve. Every parameter tensor has a finite connected gradient in
recorded actual FIT checks; all35 top-level parameter groups have nonzero
last-update gradient norms in original receipts. Full optimizer moments and
Python/NumPy/Torch RNG are saved; cold model and optimizer load before any
new Rust download. New trained weights differ from initialization; evaluation
weights remain exactly unchanged. The prospective95k–105k trainable AND deployed
parameter gates pass at actual100521, before and after fit. No historical500k
or9120p-probe waiver is reused; no dummy or inactive padding weights.

Rust Project source at Git commit2771df165f092a4fbb9f46b5b2e7450f19010adb,
MIT/Apache2.0, two chapters outside085's exposed source pair:

|New original document|Bytes|Prefit Git blob|Raw SHA256 recorded after seal|
|---|---:|---|---|
|src/ch06-01-defining-an-enum.md|15757|74e460d1e464693be3c7b9246b38d66489aa0aae|d0958df229fd25f89f140896449ce69bbd6f40db8398a99e4a53635dba0438f2|
|src/ch10-01-syntax.md|14580|8a13a252e0ef1e5b65c50be39afe6fd7bf977483|c88e9d9404ce16c820b4f4f84af73f40b4eb4978020dc3a1cead90ba0ac2569a|

Publisher Git-tree **metadata only** was read before FIT; immutable commit,
Git blob and byte lengths were preregistered. Raw content was downloaded
only after each seed's model seal. Byte length and Git content hash verified,
then additional raw SHA256 recorded. Do not claim raw SHA256 was separately
frozen before fitting. This is first new-document PSF→Rust transfer for086,
not broad independent qualification or a wholly new publisher versus085.

Each owner starts once, observes original raw0..1023 in order, and at
source t128..1023 predicts raw[t+1].896 genuine next-byte labels/owner,
1792 distinct source positions total, reused across3 models rather than5376
independent labels. No source bytes after the current t enter a prediction.
The 128-byte warmup and1024-byte stop were fixed before fitting; no selection
of favorable source clocks. New DEV acquisition/inference was performed once
per original trained seed job; later documentation/bridge/test fixes run the
launch guard and skip new fits and new DEV evaluation.

## Native graph, state and measured cost

Embedding32;3 lexical lags; learned writer8 and independent current query8
from lexical32+prior upper32. Value is normalized signed current-byte identity
plus learned bounded residual. Fast8×8 clipped residual update with learned
soft decay/admission; slow8×8 weak residual update with slower learned decay;
learned two-lane read mixture. Also active: two aligned8×8 banks, four2D waves,
48D basis mean, four8D concepts and quadratic equilibrium, fast/slow48D
processing, upper32D recurrence and94-wide SwiGLU256-way decoder. Full typed
causal graph in source preregistration and PNDS graph record. This is a scaled
new native TAC; no bitwise500k-architecture equivalence is asserted.

Per-owner tensor state **1920B**; clock **8B**; UTF8 owner identity metadata
additional and explicit. Shared learned weights **402084B**; shared fixed
byte-code buffer **8192B** (reported, not hidden in parameter/state counts).
No retained last logits and no per-owner plaintext/key index in the model.
Peak actual training process RSS roughly544–546MB; evaluation about340MB,
including framework/allocator/workspace. Tensor payload is not total process RAM.

Same payload+clock raw ring **1928B total=16B cursors+1912 data bytes**; owner
metadata excluded equally. It scans observed past contexts, so needs no stored
key table. The1024-byte trial does not saturate its1912-byte capacity. The
original evaluator key table, report buffers and diagnostics are outside the
model payload and do not affect deployed predictions.

Full per-byte model write/read/decode wall time roughly0.77–1.75ms over2048
observed steps/seed. Ring query time roughly0.93–1.65ms over1792 queries;
append work is separate. These are actual2-thread hosted CPU measurements,
not mobile latency or equal-compute architectural evidence. All diagnostic
lesion/ranking overhead is excluded from the reported full inference timer
but included in overall evaluation CPU seconds. CPU times need not match.

## Validation, engineering failures and mathematical limits

All three actual trained source jobs and strict aggregate passed. Original
checkpoint ZIP SHA digests and model-file SHA were independently reread
locally; read-only reaggregation equals original CI JSON exactly, with no new
training, source evaluation or DEV read. Full original model/optimizer/RNG,
FIT receipts, per-owner next-byte rows, query diagnostics and failures remain
in immutable source artifacts. Matched-source next-byte target hashes equal
across seeds, actual counts and frozen weights checked.

Pre-fit real-FIT suffix poison passes before first update; each original new
owner t128 future-only poison leaves actual logits and every next-state tensor
bitwise unchanged (two pairs/seed). Trained original128-prefix full/stream
max errors3.8–5.8e-6, within3e-5. State purity, source owner isolation and
no-historical-key/target API fixtures pass. The source-reviewed architecture-
agnostic probe adapter runs on FIT with actual named state and local AD;
unsupported candidate dimensions remain INCONCLUSIVE. These are scoped
engineering checks, not all-channel no-leakage certification.

Partial PNDS Lean build, Python tests, explicit axiom audit and original legacy
reciprocal-index checks passed in [40979012367](https://github.com/FPC-effortless/pnds-lean/actions/runs/40979012367)
and bridge-head [40979236896](https://github.com/FPC-effortless/pnds-lean/actions/runs/40979236896).
Actual full-model fast/slow single-step Float64 operators agree with independent
Python reference to1.39e-17; sampled local-memory derivative finite-difference
error3.11e-11. Complete learned graph Float32/AD proof, semantic task validation,
OS-separated reward/goal certification and canonical coordinated registry
integration remain OPEN. No scoped proof upgrades empirical memory capability.

The separate inherited repository-wide boundary job40979123016 initially
failed because084 and085 cached the same generic `operators` module. The
085 bridge now uses a unique module name and the **unchanged original085
operator file**;31 combined084/085/086/probe tests pass. Full local suite then
had1598 pass/4 skip and one stale canonical test-count documentation failure;
sync_test_count.py restored the current1603 count while retaining frozen327.
The relevant44 integrity/operator/model tests pass after that repair. The
085 aggregate workflow also lacked its source job's branch guard and tried
to collect absent085 artifacts on086 (run40979662207); its guard now matches
the original085 source branch. These are engineering/CI corrections, not model,
training, source, metric or hyperparameter changes. Original failures preserved.

## Research decision

Proceed only as internal evidence that an end-to-end100k native TAC can learn
cross-publisher next-byte prediction and benefit from its native read route.
Do not call this old-content recovery, slow-lane success, a clipped-memory
architecture winner or factual retrieval. The next decisive mechanism control
must isolate past state from current writes; meaningful historical selection
and source-grounded content correctness need separately registered objectives
and fresh evidence. Any trained architecture comparison still requires matched
full parameters/data/training budget; this single-family trial authorizes no
model promotion or additional tuning on these now-exposed Rust chapters.
