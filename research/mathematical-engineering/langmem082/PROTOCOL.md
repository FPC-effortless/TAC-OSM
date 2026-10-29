# LANGMEM082 preregistered compressed-value recoverability diagnostic

Question: does the original source slow 8×8 FP32 matrix expose historical
byte values through the original normalized addresses? Three original 078
checkpoints from run40890642312; original PSF source receipt/split inherited
from081; no fits, no tuning, no GPU, no protected TEST. All outcomes are
**internally measured, not independently qualified**. No model promotion.

Same four previously exposed DEV documents, observed raw0..319. At t128..319
predict genuine raw[t+1]. Historical anchors a=t-age with age8,32,128,256,
a>=4, yield2528 repeated history cells per seed, not independent samples.
768 distinct next-byte query clocks per seed. Source SHA, checkpoint SHA,
original128-prefix parity and actual slow affine identity are hard gates.

Native update (source079): M_t=M_(t-1) P_t+Q_t,
P_t=rho(I-beta w_(t-1)w_(t-1)^T), Q_t=epsilon v_t w_(t-1)^T.
All gates/value transforms frozen. Read M_t q_t (actual current query) or
M_t w_(a-1) (external evaluator supplies old address and historical clock).
Decode by cosine-nearest of all256 native_value byte encodings at clock a,
fixed ascending-byte tie rule. This uses a diagnostic untrained inverse,
not TAC's native trained output head; failure cannot prove no other decoder
could reconstruct information. No trainable probe or fitted temperature.
Codebook has256×8 FP32 shared transient entries per clock and is recomputable
from original shared weights; it is NOT per-owner persistent state.

Controls: source-derived wrong event content from the most recent earlier
byte distinct from raw[a], same historical address/clock/gates and exact
propagation, rescaled to the true latent value norm. Both true/wrong event
queries hold other prior state fixed and run actual original step before
reading the future byte. These evaluator interventions use observed old
content and cannot serve as a deployed predictor. Also zero vector; highest-cosine wrong-context writer (no future targets);
cyclic value-row permutation preserving Frobenius norm exactly; historical
event removal; isolated transported Q_a product(P_(a+1)..P_t) as evaluator
ceiling. Isolated events/old writer addresses are externally stored and NOT
included in native fixed state. One-step zero/permuted slow prestate logits
hold all other prior model state fixed. Permutations are mechanistic controls,
not naturally collected wrong-history documents. Report zero/current-value
baselines to expose byte-frequency/current-token shortcuts. No-match outcomes
remain in current-query coverage; conditional historical continuation recovery
is distinct from future next-byte accuracy.

Two packed raw rings have total256 and4160 bytes INCLUDING16B nextclock/count
metadata, hence240 and4144 actual byte capacity. No stored learned keys,
contexts or count table; query scans transiently. Return unavailable outside
retention. Report recovery availability AND accuracy, L2/3/4 Dirichlet tau8
next-byte NLL, CPU scan/decode time. Native4160B is tensor payload and excludes
owner/clock/weight identity metadata; Python allocator overhead excluded from
both. Slow component256B plus unchanged other state3904B versus ring256B
component plus3904B context is a storage allocation reference, not an integrated
trained replacement. Full4160B standalone ring remains unsaturated on320
source bytes; no inference of long-stream capacity from it.

Prediction boundary: source token producer, factors, full/lesion steps and
all ring/decoder choices run before raw[t+1] reaches separate scoring. At each
owner t128 poison all future source bytes and rerun actual token, factors,
step and every next-state tensor; require bitwise unchanged. Sham copied
slow matrix requires bitwise equal logits. Capture exact executed checks,
source/checkpoint identities, finite values and failures; broader producer,
ID-blind, gradient and full-pipeline anti-cheat dimensions remain INCONCLUSIVE
where uninstrumented. No gradients/optimizer steps in this no-fit experiment.

Acceptance is a complete honest diagnostic, including null/negative outcomes.
Falsifier: supplied-address matrix recall comparable to wrong/zero controls
while isolated event recall succeeds indicates superposition/readout loss
under this decoder. High supplied-key recall with poor native-query behavior
locates the bottleneck in access/selection/use; neither constitutes learned
factual retrieval or independent generalization. Preserve all per-owner,
per-age and per-seed rows; do not treat repeated cells as confidence samples.

Prior art reused: classical associative outer-product memory (Kohonen1972),
fast-weight associative storage (Schmidhuber1992), delta-rule linear memory
(DeltaNet, NeurIPS2024), and simple cache interpolation (Grave et al., ICLR2017).
No novelty claim for associative recovery, ring storage, or these controls.
Whole source IEEE/AD equivalence, trained semantic recovery and untouched
independent DEV generalization remain OPEN. Companion PNDS partial formulation
covers read linearity, norm-preserving permutation and packed-ring bounds only.
