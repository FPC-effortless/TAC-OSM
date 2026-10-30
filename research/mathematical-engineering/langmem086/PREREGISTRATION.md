# LANGMEM086: one complete100k compact TAC with autonomous query and clipped dual memory

Frozen before any optimizer step or new Rust content read. Internal exploratory
single-family training, not an architecture contest. Three seeds1701/1702/1703,
no tuning, no old checkpoint changes, no500k/probe-budget exceptions. All
original078 weights remain untouched. No GPU, protected TEST, promotion or merge.

## One capability question

Does a fully trained native TAC use its own bounded memory to lower actual
next-byte loss, without an evaluator supplying the historical address?085
motivates clipped recent writing but does not establish learned retrieval.
Its old supplied-key byte metric is not the primary outcome here. A practical
result requires honest source next-byte loss, correct memory use, state and
inference cost. No fresh model is compared to an unmatched trained baseline.

Prior art reused: fast-weight associative memory (Schmidhuber1992), delta-rule
linear memory/DeltaNet (NeurIPS2024), multiscale recurrent state and cache
language models (Grave et al., ICLR2017). No novelty for clipping/delta memory.
The unresolved TAC-specific question is useful **autonomous** query/write/read
under genuine source next-byte loss in the integrated concept/wave architecture.

## Actual100521-parameter graph and timing

No frozen pretrained backbone or new auxiliary readout. Every parameter is live:
256×32 embedding;7→32 positions;3×32 lexical-lag weights;32→8 learned writer;
64→8 learned query from current observed lexical context plus prior32 upper
state. Current source byte identity c(byte) is a normalized signed8-bit code;
value is normalize(c(byte)+.1 tanh(V e_t)), with32→8 learned residual.

Current observed byte t is written against **previous** learned writer k_(t-1).
Fast8×8 memory F uses base=rho_f F, rho_f=.97+.02 sigmoid(g_f0(e_t)), residual
r=v_t-base k_(t-1), eta_f=min(1,.35/(||r||+1e-9))sigmoid(g_f1(e_t));
F'=base+eta_f r k^T. Slow8×8 S uses rho_s=.995+.004 sigmoid(s), eta_s=.01
sigmoid(g_s(e_t)); S'=rho_s S+eta_s(v_t-rho_s S k)k^T. First source byte has
no previous-pair write. This is a soft learned gate with residual clipping,
not085's failed binary selective threshold. Query q_t is normalized learned
projection of observed lexical context and **prior** upper state. Learned
softmax two-lane mixture blends F'q_t and S'q_t. No caller history key/index,
source label, reward or future byte enters model forward.

Preserved compact TAC paths: two aligned8×8 time-addressed associative heads,
four2D damped waves,48D basis mean, four8D concept seeds and positive quadratic
concept equilibrium;48D fast and48D slow processing;160D representation;
32D gated upper recurrence;94-wide SwiGLU256-byte output plus concept output.
This is a scaled native TAC candidate, not a claim of bitwise equivalence to
historical500k model. No unchanged500k trained architecture comparison is made.

State tensors: F64+S64+aligned128+wave8+basis48+concept32+lexical96+previous-key8
+upper32=480 FP32 numbers=1920B per owner. Clock8B and UTF8 owner metadata
separate. No output logits persist in state. Shared learned weights402084B;
fixed byte codebook8192B also reported. This buffer is not hidden in learned
parameter counts or state budget. Transient workspace/allocator metadata
and peak process RSS reported separately. Same tensor+clock budget raw ring
uses1928B total including16B cursor/count,1912 byte payload; source owner
metadata excluded equally. No unbounded history in deployed model state.
Evaluator old-key table is diagnostic storage only and never model input.

## Genuine FIT schedule and immutable first new DEV

FIT: exact original13 PSF CPython3.13 tutorial owners plus registered Unicode
FIT supplement=14, hash/Git blob/length verified from SOURCE023. The086 loader
reads only FIT files, not old DEV. Source byte order and contiguous257-byte
windows never cross owner boundaries. `FIT_PLAN.json` freezes per-seed sample
order/raw-window SHA.160 AdamW updates, batch4,256 actual source next-byte
targets/window=163840 targets/seed; LR.001, weight decay.01, global gradclip1.
No auxiliary objective, early stopping, calibration or learning-rate selection.
Model inputs raw[:-1], labels actual raw[1:], observed-before-prediction timing.
All original prior experiments/weights/outcomes and failures are preserved.

New DEV: two Rust Project chapters outside085's two exposed chapters:
`src/ch06-01-defining-an-enum.md`, blob74e460d1e464693be3c7b9246b38d66489aa0aae,
15757B; `src/ch10-01-syntax.md`, blob8a13a252e0ef1e5b65c50be39afe6fd7bf977483,
14580B. Rust Git commit2771df165f092a4fbb9f46b5b2e7450f19010adb, MIT/Apache2.0.
Metadata comes from the publisher's immutable Git tree, with **no raw content
read before fit**. Git blob/byte size pinned before fitting; raw SHA256 is
recorded after checkpoint seal and verified Git content acquisition. Do NOT
claim an independently frozen raw SHA256 before fit. The original blob pin
is the prefit source-content contract, not an invented SHA256 authority.

Only after each seed's160 updates, full model/optimizer/Python NumPy Torch RNG
checkpoint, SHA seal, actual final size gate and bitwise cold replay on FIT
prefix may that job download the two new Rust documents. Validate Git blob
and length before any source inference. This is first new-document cross-
publisher transfer from PSF FIT to Rust, not a large independent qualification
suite. Seeds share documents and source targets. All three seeds mandatory,
no adapting later fits to early seed DEV outcomes. Workflow hashes fixed.

## Primary outcomes, controls and falsifiers

Each source owner is reset once, then byte0..1023 observed consecutively.
Score t128..1023 against genuine next raw[t+1]:1792 actual next-byte targets
per seed, source owner/clock/target SHA equal across all three models.
Report NLL/top1 by owner and seed, plus read-intervention NLL differences.
No significance computed over repeated seed/document bytes as independent
samples. New data/clock exclusions cannot be selected after outcomes.

Controls are **within the same trained model**, at identical original prior
state/current observation: zero fast read, zero slow read, zero both native
reads, and norm-preserving cyclic value-row permutation of both read matrices.
Memory writes and all other prior model state stay active; lesions are not
full historical state erasure. No language architecture-superiority claim.
Compare FIT-only byte prior built from **exactly the same sampled FIT targets**,
and same-budget raw ring with four-observed-byte context counts, tau8 shrinkage
to that FIT prior. No prior fitting on Rust or extra all-corpus exposure.

Every8 source clocks, rank external logged old writer addresses using actual
model q_t over past age<=256 and report exact observed four-byte context match
coverage/top1 when available. This evaluator bookkeeping is not fed back into
inference and is not factual semantics. A selected past continuation is an
external reference, not a model-decoded historical byte. No autonomous
factual-retrieval victory can be inferred from byte-context rank alone.

Acceptance is complete honest learning/use evidence. If removing slow/fast
reads improves loss, or if learned model does not beat simple FIT/ring
references, preserve it as failure to demonstrate the intended usefulness.
A small single-model lesion effect cannot establish clipping superiority;
that needs a separate fully matched trained comparison and fresh DEV. Never
choose a model/seed or retune after seeing these outcomes.

## Gates and explicitly OPEN properties

Actual live counts before first update and after training must equal100521,
both trainable/deployed in95k–105k; no dummy/unused weights. Source and sample
order hashes, checkpoint optimizer/RNG loading, cold logits replay, finite
connected gradients for every parameter and per-module magnitudes recorded.
Actual new-source streaming/bulk128-prefix outputs within3e-5; future-only
source poison at owner t128 must leave next logits and all next-state tensors
bitwise unchanged. Constructed full-forward future suffix poison, state purity,
owner isolation and no-historical-key/target API checks must pass before fit.
Register/run actual ResearchProbeAdapter for model state/read/local AD on FIT,
with unsupported candidate and uninstrumented graph dimensions INCONCLUSIVE.

PNDS record includes full typed input→write/read→concept/upper→prediction/loss
spec, exact-real clipped-update/mixture and dimension/count contracts, Python
operator reference and source numerical tests. Complete graph IEEE/AD proof,
OS-separated evaluator/reward certification, semantic/factual goal evidence,
all-channel no-leakage clearance and model promotion remain OPEN. Ordinary
internal research needs no invented external signer. No protected TEST access.

Route: source-controlled phone planner chooses GitHub Actions CPU, estimated
45 CPU minutes over3 seeds, maximum45 wall minutes per job. An actual prefit
4×256 forward/backward profile took2.38 CPU seconds with zero optimizer updates;
160 steps plus bounded evaluation fits this conservative CPU budget. Zero GPU
quota; no Kaggle fallback or accelerator use is permitted.

To prevent report/bridge commits from silently refitting models or rereading
fresh DEV, source workflow launches training only on reviewed commits whose
subject begins `LANGMEM086-TRAIN:` on this exact owner branch. Normal later
commits run the no-fit guard and skip training/evaluation. This marker does
not bypass any preflight/model/source gate. If a reporter fails after actual
fits, recover sealed artifacts read-only; do not retrain to fix reporting.
