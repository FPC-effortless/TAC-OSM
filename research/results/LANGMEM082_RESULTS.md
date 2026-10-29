# LANGMEM082 — compressed value readout lacks clear historical address specificity

**Completed internal measurement, not independently qualified.** Original
three-seed CPU source run [40962739303](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40962739303)
passed all three model jobs and strict same-source aggregation. Immutable
[aggregate artifact12935763268](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40962739303/artifacts/12935763268).
Source implementation **dc39eaaa7f7cb3209173f7117bab7daa503f84b7**, draft
[TAC#624](https://github.com/FPC-effortless/TAC-OSM/pull/624).
Partial Lean build, axiom inspection, Python reference tests and legacy
reciprocal-manifest checks passed in [PNDS40962743965](https://github.com/FPC-effortless/pnds-lean/actions/runs/40962743965),
[draft PNDS#79](https://github.com/FPC-effortless/pnds-lean/pull/79).
Original complete aggregate retained verbatim semantically in
`LANGMEM082_AGGREGATE.json`; subsequent equal-denominator paired reaggregation
of the SAME three original artifacts in `LANGMEM082_PAIRED_AGGREGATE.json`.
No results replaced, no fits or checkpoint changes.
Final source replay [40963357959](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40963357959)
on **a6601b6e720a97c659980d2d56e4dfcb0e27a96b** passed all three
model jobs, CI-pinned PNDS operator parity and equal-denominator paired
aggregation. All byte-recovery accuracies and source target SHA are identical to the
first run. Aggregate floating-point NLL values differ by at most2.17e-9;
this is not bitwise replay identity. See `LANGMEM082_REPLAY_CHECK.json`. Current PNDS bridge head also passed in
[40962992755](https://github.com/FPC-effortless/pnds-lean/actions/runs/40962992755).

## Frozen genuine-source cohort and decoder

Same three immutable original LANGMEM078 checkpoints with 499,627 parameters from
run40890642312, full optimizer/RNG/source-order receipts, original079 causal
4160B carried state, same four genuine licensed PSF CPython3.13 already-exposed
DEV owners raw0..319. Original trained model NLL **2.1202219517** matches081.
Per seed:768 genuine next-byte targets,2528 historical query cells at ages
8/32/128/256 with a=t-age>=4. Counts768/768/752/240 per age; repeated source
cells and labels are not independent samples. No held-out independence claim.

The fixed diagnostic reads actual slow M_t w_(a-1) using an **evaluator-supplied
old writer address and clock**. It cosine-decodes against all256 original
native_value byte encodings at that historical clock. This is an untrained
inverse probe, **not TAC's learned decoder or an autonomous event selector**.
External original key/event stores never count as native4160B memory. Failure
of this decoder is not a proof that no decoder could recover any information.

## Correct addresses do not clearly improve byte recovery over wrong addresses

Mean across three original trained seeds; percentages score raw[a]. Wrong
address is the highest-cosine earlier writer with a different observed
four-byte context, label-blind to future raw[t+1].

| Historical age | Correct old address, actual compressed matrix | Wrong-context address, same matrix | Historical event removed | Isolated historical event only | 256B raw ring available/correct |
|---|---:|---:|---:|---:|---:|
|8|5.425%|5.729%|3.906%|100%|100% / 100%|
|32|10.981%|10.938%|9.809%|100%|100% / 100%|
|128|24.025%|24.113%|23.670%|100%|100% / 100%|
|256|8.333%|8.611%|8.194%|100%|0% / unavailable|

The isolated event ceiling is computed using an **external evaluator**
decomposition Q_a P_(a+1)..P_t and its old key, and excludes interfering
writes. Its100% means the rank-one value direction can be inverted in this
privileged setting, not that compressed native memory reads it successfully.
The true matrix percentages and positive latent cosine alone cannot establish
historical content specificity: wrong addresses give essentially the same
accuracy, and removal of the target event often barely changes recovery.
A post-run descriptive label-frequency check (not a preregistered predictor)
finds that byte42 (`*`, publisher header text) alone accounts for23.936%
of the age128 historical labels. Thus24.025% matrix accuracy is approximately
the cohort modal-byte frequency; it is especially poor evidence of recall.
Row-permuted content yields0.28–0.44%; zero read always predicts byte0 and
scores0%. These are weaker controls than same-source wrong keys/event swaps.

## Rigorously paired correct/wrong historical content

The wrong event uses the most recent earlier source byte differing from
raw[a], re-encoded at **the same clock a**, scaled to the true value norm,
then uses **the same historical writer key, gates and propagation matrices**.
It replaces only that event in the actual source slow state; other prior
model state is held fixed. The resulting source step is checked against the
expected counterfactual matrix within1e-5. Mean event norm errors are below
3e-10. This is an evaluator intervention, not a naturally occurring alternate
publisher document or a deployable predictor.

Where no earlier distinct-byte donor exists, the pair is unavailable. The
following uses **exactly the same valid rows in both arms**, unlike comparing
all-row matrix accuracy with a conditional wrong-event average:

|Age|Valid pairs/seed|Correct event|Norm-matched wrong event|Wrong minus full next-byte NLL|
|---|---:|---:|---:|---:|
|8|768|5.425%|3.082%|−3.19e−6|
|32|768|10.981%|9.375%|−1.57e−7|
|128|748|24.153%|23.619%|+6.57e−8|
|256|236|8.475%|8.333%|+7.16e−8|

There is a modest short-age content contribution under this inverse, while
correct versus wrong content has very small mean next-byte effects and tiny
long-age recovery differences. Do not interpret these dependent cells as
statistically independent evidence or the long-age24% as semantic recall.

## Actual current-query read and storage allocation reference

Using the model's **actual current native query q_t**, the slow matrix's
fixed inverse predicts the real next byte on **2.821%** of768 queries.
On the17.448% of queries with a past exact four-byte context in H256, it
recovers the most recent matching **historical** continuation on **9.701%**.
The corresponding actual stored historical byte equals the future target on
74.627% of those queries. This separates failure to read the observed old
value from the independent ambiguity of historical continuation.

Original trained full model NLL2.1202219517; one-step zero-slow NLL2.1203107143;
row-permuted-slow NLL2.1203311593. Those tiny aggregate effects do not imply
that no source individual prediction is affected or that the full model's
other memory paths lack useful information.

Packed raw ring totals **256B=16B metadata+240B bytes** and **4160B=16B+4144B**.
At registered ages<=128 both retrieve retained bytes exactly; the256B ring
correctly refuses age256. Full4160B ring is100% here but **unsaturated on320
bytes**. The256B ring standalone L2/tau8 next-byte count predictor NLL is
**4.6655986472**, much worse than the full neural2.1202; retaining plaintext
is not enough to replace model predictions. L3/L4 NLL4.91445/5.06193.
The4160B ring with same H256 reproduces081 L2 NLL4.6666562779.

This compares persistent storage allocation, **not a trained architecture
winner or an integrated ring replacement**: native4160B is tensor payload
(excluding owner/clock/signature metadata), whereas rings include16B cursors;
Python object/allocator overhead excluded for both. Other native tensors total
3904B; attaching them to a256B ring would consume4160B payload but is not a
trained/evaluated graph in this study. Evaluator keys/events and codebooks
are outside native persistent memory. Timings are saved per seed; matrix
selector/decode and ring scan have different work units, not an equal-work
latency benchmark or mobile measurement.

## Executed verification and remaining unknowns

All three source checkpoint identities/fit-order receipts preserved; actual
slow affine update and original128-prefix parity passed. Each seed passed
four owner t128 whole-future-byte poison pairs and four bitwise copied-state
shams; model weights unchanged.36 constructed engineering/probe tests pass.
Pinned PNDS Float64 operator parity passes at1e-12 tolerance, maximum read
error1.776e−15; Python mathematical tests and Lean theorem/axiom checks pass.
These are partial checks, not a global no-leakage or whole-model proof.

No optimizers/gradients/new model training, GPU, protected TEST or merges.
Full source IEEE/AD, unseen independent DEV, autonomous historical address
selection, alternate trained decoders and whole-history content interventions
remain OPEN. Legacy canonical two-repository registry merge also OPEN;
source-pinned per-study reciprocal bridge is recorded separately.

Next work should target learned selective writing/reading/decoding and
longer authentic FIT context with independently registered unseen DEV, within
the prospective100k parameter and matched-data/training-budget rules. This
experiment does not authorize training or open TEST.
