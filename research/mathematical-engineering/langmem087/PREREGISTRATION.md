# LANGMEM087 staged slow-memory learning study — stage A only

Internal research, not independently qualified. Frozen before any087 optimizer
update or new DEV raw-content access. No architecture, matrix-size, write-cap,
read-bias, dropout, sequence-length, loss or optimizer change in stage A.

## Questions and staged scope

User's hypotheses are credible alternatives: insufficient training, little
long-context pressure, weak slow writes, mixture preference, and query/write
alignment. Reuse fast-weight/delta memory (DeltaNet NeurIPS2024), multiscale
memory and cache-language-model prior art from086; no novelty for these ideas.
The unresolved goal is useful autonomous access to genuinely observed history.

A executes now: original three086 seeds1701/1702/1703 at160updates continued
unchanged to640 then1280; all three durations measured prospectively on the
SAME two fresh documents. This preserves baseline checkpoints exactly and
avoids training a new160 baseline whose trajectory might differ. The duration
contrast intentionally changes data exposure and optimization budget; it is
not an equal-budget architecture comparison. No checkpoint selected/tuned.

B is designed but NOT launched here:256/512/1024 training lengths should use
one common1025-byte owner-local source span per update, split as4x256,
2x512 or1x1024 adjacent label windows. All arms expose exactly the same1024
next-byte targets in source order and use identical update counts. Register
fresh data/order, measured FLOPs, reset semantics and 3seed matched parameters
before fit; credit horizon, state-reset frequency and batch-size effects must
be distinguished. No all-history benefit claimed from longer evaluation.

C is designed but NOT launched here: initialize the existing slow output bias
in read_mix.bias[1] at its original sampled value plus0/+1/+2. It is already
trainable: NO extra bias parameter, no constant value addition. From matched
initialization use identical data/order/steps and count100521. Record actual
mixtures, gradients, NLL and old-state controls, not illustrative equal-logit
weights as measured outcomes. A separate fully matched fast-only trained arm
is needed for architectural usefulness, not just read lesions. Register fresh
DEV before fitting. No choice of favorable seed or opened087document tuning.

D0.01vs0.05 cap, E genuinely observed history supervision, occasional fast-read
dropout, and read-before-write matched retraining remain later registrations.
They are not bundled into this duration test. Historical labels must never
be evaluator-supplied inference keys. No synthetic task is primary evidence.

## Stage A exact fit and replay

Original086 checkpoints/optimizer/RNG/archive pins in PARENTS.json. Original
compact_tac.py SHA7179908fa3672879f3c2046ec4521f0b9f783d5c696aaba82c4c2cd0b6ebc9b1,
100521trainable/deployed active parameters, 1920B persistent tensor state,
8B clock + ownerUTF8 metadata,402084B shared learned weights,8192B fixed buffer.
No original078/086checkpoint is modified.

Original14PSF CPython FIT sources/blob/SHA, exact086 sampler np.default_rng
perseed,4x256, owner-local contiguous257byte windows; regenerate and skip
first160 then continue. Source and windows at160/640/1280 hashes frozen in
FIT_PLAN.json. LR.001 AdamW wd.01 globalgradclip1 all unchanged; no schedule,
auxiliary objective or earlystop.160=163840,640=655360,1280=1310720targets/seed.
The first two independent continuation jobs480/640 updates preserve every
parent weight and optimizer step counter, checkpoint fullmodel/optimizer/
Torch/Python/fullNumPy RNG, cold bitwise logits/state and optimizerstep replay.
Original086 stored only unused globalNumPy key-array: reconstruct original
seed state and verify it; actual sampler uses its own regenerateddefault_rng.
All nine checkpoints must finish before ANY087 DEV acquisition in workflow.
No outcomes are used to adapt remaining seeds/updates. Save failures too.

Run phone planner separately for each bounded per-seed CPU continuation and
evaluation job. Per-leg estimates25/35minutes (maximum45wall/job), total
additional fit budget180CPUminutes across3seeds conservatively, plus75CPU
minutes for3evaluation jobs; actual CPU/latency/RSS reported. These are job
boundaries with real checkpoints, not a GPU fallback or quota consumption.
Original4x256 forward/backward profile repeated without optimizer before fit.

## New source content and timing

Rust book publisher commit2771df165f092a4fbb9f46b5b2e7450f19010adb, MIT/Apache2.
Two previously unused paths ch07-01-packages-and-crates.md (3806B Gitblob
63e6d4142655c9601083608ebb31655aece3303d), and
ch09-01-unrecoverable-errors-with-panic.md (8520B Gitblob
f74ad3206c0ece21e235117561c217ca9dfb590a). Only Git-tree metadata inspected
before fitting. Raw SHA256 recorded after all model seals, Gitblob/size
verified against frozen registry. No claim of independently frozen raw SHA.
Same publisher as085/086, different chapters: limited new-document transfer,
not broad external/semantic qualification. TEST remains unopened.

For every checkpoint: reset each source, observe raw[0:2048], score t256..2047
against actual raw[t+1].3584 distinct source positions per checkpoint, repeated
across durations and seeds; NOT32256 independent observations. All predictions
and controls precede revealing next-byte labels to scorer. Prior/caches use
only observedprefix and exactly exposed FITtargets. No adaptation or refit.

Primary: NLL/accuracy byseed/owner/duration, paired640−160 and1280−160 changes.
Report fixed streamclock strata256..511/512..1023/1024..2047, NOT historical
content ages. Report slow mixture, gain, finite connected module gradients.
No checkpoint selection. Lower overall loss without historical-content
specificity is language-learning evidence only. Two owners are insufficient
for broad population significance; no iid byte/seed pseudo-replication.

## Old-content versus current-write attribution

For frozen original prior state, currentobservedbyte, originalk,v,rho,eta,q,mix:
S_new=rho*S_old(I−eta*k*k^T)+eta*v*k^T = H+C.
One-query controls read: fullH+C, old_slow_zero=C, current_slow_write_zero=H,
old_slow_permuted=rho*P*S_old(I−eta*k*k^T)+C (cyclicvalue-rowpermutation),
alongside086fast_zero/slow_zero/both_zero/permuted_read. Original stored next
slowmatrix and other nonupperstate remain exactly unchanged. Downstream
upperupdate changes with read, then is discarded: next evaluated query always
uses original trajectory. These are immediate effects, not persistent erased
history runs or separately trained controls. Default graph must be bitwise
original086 before using any attribution. SplitFloat32 tolerance1e−6.

A positive old_slow_zero−fullNLL removes all ages>=one previousobservation;
it is evidence of dependence on older matrix content under this one-query
intervention. It does NOT establish retrieval of a particular distant byte,
causal semantic recall, or superiority to a matched trained fast-only model.
Oldstate permutation tests value-coordinate sensitivity but is an OOD lesion.
Other history-carrying states and currentk/query themselves still containpast
information. No claim that one-step lesion clears every historical channel.

Model's current query must be autonomous, never evaluator keys or labels.
Every16clocks diagnostic ranks logged writerkeys atages<=256 with exact4byte
context labels. Sign/rotation/distributed coding can invalidate nearest-key
proxy as a recall test;0doesnotproveabsence. No historical-byte decoder head
or semantic answer retrieval has been added.

FIT byte prior and1928B ring from086 retained, ring1912Bpayload+16Bheader,
shared256float64prior2048B separate; learnedweights/buffer/prior/workspace/RSS
not hidden. Ring saturated in late stream. Model observed-step latency vs
ringquerylatency have different units/work; no equalcompute efficiency claim.

## Gates and falsifiers

Actual100521counts before each continuation/finalcheckpoint, exact parent
SHA/source/samplerprefix/optimizerstep, finite connected gradients everyupdate,
FIT future suffix poison, cold fullcheckpoint replay and actual086adapter
state/localAD probes; tests verify ordinarygraphidentity, oldsplit/currentterm
and immutable priorstate, samplerprefix. Actual newsource bulk128/streaming
within3e−5, futureonlysuffixpoison at256 bitwise prediction/allstate, frozen
weights and complete cohort/source identity. Missing unsupported probe
dimensions INCONCLUSIVE. External reward/goal/ACL certification OPEN; no
promotion. All quantitative source outcomes retained even if hypothesis fails.

Lean/Python scoped old/current decomposition + original086operators, actual
PyTorch source numerical/gradient checks mandatory. Full graph IEEE/AD proof,
semantic recall, external qualification, canonical registry merge remain OPEN.
Normal report/bridge commits MUST NOT refit/reread: exact ownerbranch launch
only when reviewed subject begins LANGMEM087-TRAIN:. Reporter failures recover
original artifacts read-only; never retrain to repair aggregation.
