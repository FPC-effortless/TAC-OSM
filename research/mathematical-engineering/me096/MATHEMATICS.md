# ME096 shared-core evidence retrieval: prospective capability development

Lane: **physical**, one candidate within the existing six-family shared graph.
No seventh family; no independent heads/skills for the other five families.
Parents: TAC #390 ec0db237875cf5fd0c6a6293ae0b01489d31a284 and #389.
Read master plan a6f0424c before this iteration. ME095 interrupted originals are
preserved in separate research PR #377, commit b30b5aaf; not scored or retried.

## Input to output and legal parallelism

Inputs: finite normalized observed x[B,T,14], previous-frame recorded action
u[B,T,14], bool valid[B,T], optional carried state (sum[B,88], count[B,1],
M[B,8,8]). Identity comes from caller-assigned stream; no inferred identities.
Valid masks are omitted/padded updates, not a learned missing-value estimator.
No future commands, observations, episode ID, remaining duration or TEST outcomes.
Input normalization and residual RMS scale come only from FIT observations.
Publisher does not establish physical command issue ordering: using u[t-1]
avoids the same-frame issue; this remains logged conditional forecasting,
not identification of an intervention or a safe controller.

1. e=tanh(Encoder[x;u]) [48], unit-ball write k=normalize(Address(e)),
   v=normalize(Content(e)) [8]. Initial basis b=tanh(Fast[e;0;0]) [88].
2. Exclusive valid prefix mean c=(sum0+sum(valid*b) before t)/max(count,1).
   Same original provisional native fast/slow mappings produce pf,ps [88 each].
3. Original controlled gates: raw a,b=sigmoid(G[e;pf;ps]); beta=b/4,
   alpha=(1-.02*(1-a))*(1-beta/4). Actual code preserves parent's
   sigmoid(logit()) stages and clamp; independent reference matches these.
4. P=alpha(I-beta kk^T), Q=beta vk^T. M'=M P+Q.
   Invalid rows set (P,Q)=(I,0), evidence/count increments zero.
5. New retrieval query q=normalize(k+Query[e;pf;ps]);
   r=M_previous q reads BEFORE the current write. Normalization controls scale;
   it DOES NOT force correct key-query alignment.
6. Read-refined original fast/slow representations f,s. New direct retrieved
   correction D(r)=Linear(SiLU(Linear(r))) [8->32->14].
   prediction=x+delta_scale*tanh(Readout[f;s;r;e]+D(r)).

Independent query and decoder cannot change an already scheduled update.
All schedule inputs are known before M reads, so balanced affine scan retains
O(T) fixed-size-matrix work / O(log T) temporal depth. Independent encoder/query/
decode operations vectorize across batch/time. This is not a measured GPU speed
claim. Streaming calls T=1; each entity retains 152 FP32 values (608 bytes) plus
one int64 count (8 bytes), excluding model/activation/runtime storage. The sum
is not itself bounded indefinitely; the existing radius-four memory argument
is conditional on unit-ball keys/values and bounded initial M, not useful recall.
All-history mean and fixed finite representation can still lose rare information.

## Mathematical status

PNDS `PNDS/EvidenceRetrieval.lean` defines the typed observation-to-output graph,
proves padding state identity, read-before-write and query nonmutation with no
axioms. Maps stand for concrete tensor operations; **full float neural refinement
and global semantic correctness are OPEN**. Existing PNDS RetentionBudget theorem
covers only the inherited controlled memory assumptions. The new independent
`foundation/v1/evidence_retrieval/reference.py` reconstructs EVERY neural map from
source weights and executes serial rank-one updates, with FP32/FP64 predictions,
carried state, input gradients and all parameter-tensor gradients checked against
actual TAC scan source (rtol/atol 1e-5/1e-6 FP32, 1e-8/1e-10 FP64). Same-graph
chunk/scan tests include causal prefix, invalid identity and nonzero gradients.
These are computation checks, not learned skill or whole-network Lean proof.

## Original real training protocol (frozen before fitting)

One 95,874 trainable/deployed parameter candidate, five original seeds 96001–5;
no padding weights, frozen encoders or out-of-band learned modules. Guarded
optimizer copied unchanged from TAC f1d5fd08 `parameter_budget.py`; measured live
before every update and final checkpoint. 256 AdamW updates each, batch8,64frames,
131,072 supervised transition exposures per model, repeated windows rather than
independent new records. No curriculum/sweep/retry after seeing scores.

Source: lerobot/aloha_mobile_cabinet, pinned revision and parquet SHA in registration;
publisher Apache-2.0 README preserved with original bytes. 59 FIT episodes,
472 windows;13 previously exposed DEV episodes,104 windows. Last15% not scored.
One-step standardized MSE, late-half MSE, and movement-rich MSE (threshold FIT90th
percentile) are descriptive measured abilities, not arbitrary-unit cross-task gains.
EVERY seed must reduce whole/late MSE by5% vs persistence and not worsen moving
MSE. All originals committed before scoring; independent original replay required.
A pass qualifies only this exposed-DEV development gate. It cannot repair old
ME094/095 gates or qualify external-site generalization or architecture superiority.
No rival fits, equal-compute superiority or per-component attribution claims.

## Prior art and capability-to-value card

Reuse gated delta error correction and input-conditioned scan algebra from Gated
Delta Networks (ICLR2025, https://arxiv.org/abs/2412.06464), and work/depth separation
from Mamba-2 (ICML2024, https://proceedings.mlr.press/v235/dao24a.html).
Separate learned read/write addresses are established memory-network practice;
combination is a TAC engineering hypothesis, not invented mathematical novelty.
Question: can a bounded retained associative representation inform real next-state
corrections through a better-conditioned retrieval/output pathway?
Potential operator use: read-only short-term prediction from verified telemetry.
External customer, intervention utility, autonomous policy and ROI: NOT_MEASURED.
Original Git weights/optimizer/RNG/schedule/preprocessing and independent replay
are compulsory. No merge, deployment or scientific promotion follows from tests.
