# DIAGGRAD001 — actual gradient flow, not a new fit

All 15 original ME106 checkpoints were traced on authentic 2013–2014 UCI501 FIT
data. No model, optimizer moment, gate or prediction was changed. No REPORT or
protected 2016–2017 numerical data were loaded. Registration and diagnostic code
were committed at `2d5edb8c` before the original run. The unchanged model source
and checkpoints descend from ME106 `e6ac736a` / draft PR408.

**Answer:** there is measurable negative gradient alignment, concentrated in
interval-score versus point prediction on these probes. The simple hypothesis
that NLL generally fights point accuracy is not supported here. The existing
global projection does not protect every shared block or the AdamW displacement.
These are local, post-training, FIT-conditional diagnostics, not proof of the
cause of failed qualification or of a successful remedy.

## Complete active path

1. Input `[B,T,13]` measurements are zeroed where unobserved. Masks, causal ages,
   six clock features and a four-dimensional station embedding form 49 encoder
   inputs. The encoder produces 48 event features.
2. Learned admission gates update 13 source sums and masses by a diagonal affine
   scan. Missing readings cannot write numeric source evidence. A 62-dimensional
   feature sum/count supplies all-history evidence. Fast/slow proposal maps
   produce 88-dimensional representations; they are not this variant's complete
   persistent state. Keys, contents and input-conditioned gates drive the 8x8
   affine associative scan. No decoder-to-same-scan feedback is introduced.
3. The original future query reads **pre-current-event** memory. Reused fast/slow
   maps and future decoder produce six standardized-log locations and scales.
   Observed last values or source estimates anchor the forecast.
4. The reconstruction branch instead reads **inclusive** post-current memory with
   the shared write-key projection. Its 102 features contain source means/masses,
   evidence means, ages and retrieval. The current-state mean/variance correct the
   missing-channel future forecast; the affine response is not the full-network
   Jacobian. Reconstruction is supervised only on genuinely recorded FIT readings
   deliberately hidden by the original augmentation.
5. The same 102 features feed left/right width and median-correction heads. Width
   factors lie in `(1/4,4)`; hard source/missingness floors then protect scales.
   The median correction is bounded by 0.5 standardized-log units. A coherent
   censored equal-half Gaussian law in log1p coordinates yields raw quantiles,
   a positive-density NLL and the atom at zero. This is one six-hour marginal
   prediction, not an action policy or joint multi-horizon trajectory.
6. ME106's uncertainty view cancels the direct median-head tangent while retaining
   exact forward values. Point and location losses keep the live median path.
   All five objectives still interact in shared ancestors. Global gradient
   projection precedes clipping and AdamW's two learning-rate groups.

Persistent state, parallel affine scheduling, parameter count (100,349), source
floors and forward/probability equations remain unchanged. Existing PNDS PR19
defines the inherited graph and limited exact projection theorem; whole-neural,
Gaussian-law and IEEE refinement remain OPEN.

## Registered measurement panel

15 frozen models x 3 contexts (8/64/256 hours) x 2 origin populations x 2 batches
= **180 gradient probes**. Batch size16. Same selected origins and artificial
missingness across recipes/seeds. Natural-gap origins mean PM2.5 is naturally
stale; they do not mean every channel is absent. General origins can also contain
natural gaps. The original augmentation may additionally hide recorded values.

All five raw objective gradients, coefficient-weighted norms and ten pairwise
cosines are measured in 13 exhaustive parameter blocks. Zero-norm cosines are
undefined/null. Additional point/NLL cosines stratify supervised targets by the
corresponding last-input channel's observation status, not target missingness.
No statistical independence is assigned to repeated
origins, the same batches across seeds, overlapping windows or warm-start lineages.

## Observed conflicts and magnitudes

Whole-model results:

| Pair | Negative / nonzero probes | Median cosine |
| --- | ---: | ---: |
| NLL vs point | 0/180 | +0.6290 |
| Point vs interval | 139/180 | -0.3297 |
| NLL vs interval | 88/180 | +0.0140 |
| Point vs reconstruction | 27/165 | +0.2470 |
| Reconstruction vs interval | 92/165 | -0.0857 |

Point/interval opposition occurs in 51/90 general-origin probes and **88/90
natural-gap-origin probes**; median cosines -0.0358 and -0.5785 respectively.
NLL/point whole-model median cosines by history are +0.6394/+0.6511/+0.5436;
none is negative. Separating observed and missing target channels likewise yields
0/180 negative NLL/point cosines in either subset (medians +0.5891/+0.7132).
Positive pooled alignment does not imply every layer agrees.

Median coefficient-weighted whole-model gradient norms across all probes:
interval7.1097, NLL2.1921, point2.1236, reconstruction1.0714, log-location0.2322.
These are descriptive pooled medians across three point weights, not norms at a
single shared model. The interval objective is often the strongest local signal.

Shared-block conflicts are real: point/interval is negative in content149/180,
encoder145/180, reconstruction143/180 and slow-map138/180. Point/reconstruction
is negative in the slow map109/165, but its median weighted reconstruction norm
there is only0.00150 versus point0.09656: angle alone overstates its magnitude.
The future query receives **zero** direct reconstruction gradient by construction;
inclusive reconstruction uses the shared write key rather than that query.

## Where ME106's protection stops

Point versus combined auxiliary dot is negative before projection in96/180
whole-model probes. After projection its worst global cosine is -6.59e-9,
consistent with small array-rounding residuals. The raw JSON's thresholded
`opposing_aux_after=48` (dot < -1e-10; 49 dots are strictly negative) is
**not 48 substantive failures**: those global residuals
are tiny, and no global cosine is below -1e-6. Do not interpret a fixed absolute
dot threshold as a numerically robust proof failure.

In contrast, substantial blockwise auxiliary opposition remains: content66/180,
slow54/180, query27/180, reconstruction76/180, belief81/180 (cosine below -1e-6).
Whole-vector orthogonality can coexist with opposing sub-blocks.

A read-only hypothetical next AdamW direction uses each original checkpoint's
saved moments, step1536, original clipping, group rates, epsilon and decay. The
dot of the current weighted point gradient with this displacement is positive
(locally uphill) in **34/180** whole-model probes: general4/90, natural-gap30/90.
By history:10/60,11/60,13/60. Group-rate SGD alone is uphill in0/180 globally on
these probes, so the mathematical possibility of group-rate reversal is **not**
observed globally here. Momentum/preconditioning/decay are combined in the shadow
AdamW calculation; this panel does not isolate which contributes.

This is a FP64 algebraic forecast of a step, tested against PyTorch on toy
parameters. No real model optimizer was stepped. It is not a reproduction of
historical training batches, measured finite-step point loss, an attribution of
the entire qualification failure, or evidence that AdamW is generally harmful.

## Decoder coupling and inactive paths

Across repeated probes, 11,175 currently missing channel slots are inspected.
The left scale equals its protected floor in7,549 (67.55%), right in10,270
(91.90%). These are repeated slots across models, not distinct sensor outages.
The suppressed learned-width branch has zero derivative where a hard floor wins.
Relaxing those floors would repeat ME104's uncertainty collapse and is not an
authorized remedy. Median saturation near its correction bound is rare here:
11 out of17,280 total predicted channel slots.

Away from censoring, raw interval width is
`exp(mean + scale_target*(mu+z*s_right)) -
 exp(mean + scale_target*(mu-z*s_left))`, with z=1.64485.
Its derivative in mu is `scale_target * width > 0`. Thus a width penalty can
prefer lowering location when a scale floor blocks shrinking scale. Miss penalties
can oppose this; the full proper interval score remains a valid scoring rule.
This derivative explains a possible coupling mechanism, not proof that every
observed conflict is generated by a floor or a defective objective.

## Engineering decision

Do not add a universal TAC/OSM loss split or another point-weight sweep. The
measured conflict is more specific. A new prospective training experiment should
retain the protected predictive law and parallel core, and test **blockwise,
optimizer-aware routing of the proposed displacement**, rather than relying on a
global Euclidean gradient projection. Protect point-loss directions in shared
blocks while leaving dedicated width coordinates available for proper likelihood
and interval training; do not suppress all reconstruction learning.

One candidate correction on a block is
`delta_safe = delta - max(0, <g_point,delta>)/||g_point||^2 * g_point`
when the point norm is positive. It bounds that block's first-order point dot,
including a proposed preconditioned displacement, but changes the optimizer;
clipping, zero-norm handling, decay and moment semantics must be explicitly
registered and independently verified. It gives no finite-step, uncertainty or
qualification guarantee. A richer source-conditioned distribution remains a
separate architecture hypothesis if routing cannot resolve the trade-off.

No candidate is trained by this diagnostic. Any follow-up requires fresh
registration, live100k checks, original matched data/exposure/compute when making
a comparison, all five lineages and unchanged evaluation gates. ME106 remains a
failed qualification; no model has been promoted.
