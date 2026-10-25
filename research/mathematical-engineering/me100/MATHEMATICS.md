# ME100 — source-aligned entity memory and a coherent sensor forecast

This is a new single-model capability pilot within TAC-Temporal, not a seventh family or a matched Phase2 architecture qualification. It implements the first source/admission step of draft PR394's direction. It does not implement concept-swarm CED, mHC, Muon, spatial maps, action selection or delayed-arrival authority handling.

## Complete computational graph

A source station supplies 13 encoded channels (11 numerical columns plus wind sine/cosine), 13 observation bits, 13 causal staleness features, six calendar features and a four-dimensional station embedding. A 49→48 tanh encoder produces e_t. Missing numerical inputs are zeroed inside the model, irrespective of caller values. Observed standardized inputs are clipped to [-8,8] using FIT-only statistics; pollutant channels use log1p first.

Admission is beta_t = mask_t * sigmoid(W e_t+b). Each channel keeps a normalized admitted sum S and admitted mass C:

S_t=(1-beta_t) S_(t-1)+beta_t z_t/8; C_t=(1-beta_t) C_(t-1)+beta_t.

Starting at zero, |S|<=C<=1. Missing channels have beta=0 and preserve both coordinates exactly. Normalized S/max(C,1e-6) remains bounded. A separate 62-dimensional tanh evidence basis is cumulatively summed and divided by the count of events containing any observed channel. Thus [normalized S13, C13, mean62] is the 88-dimensional context. All these contexts are EXCLUSIVE of the current event when scheduling the native associative transition. The final physical anchor may use the inclusive trace.

The inherited fast/slow input-conditioned maps schedule native controlled-retention delta memory. With unit-normalized k,v, b in (0,1), alpha=sigmoid(a)*(1-b/4),

M_t=alpha_t M_(t-1) + b_t (v_t-alpha_t M_(t-1) k_t) k_t^T.

Events with no observed channel apply the identity matrix-memory update. This preserves the existing bounded-memory law. The production implementation composes P_t=alpha_t(I-b_t k_t k_t^T), Q_t=b_t v_t k_t^T as chronological affine operators. Gates depend on encoded observations and prefix evidence, never on a read from this same matrix scan. Final retrieval reads M_(T-1), then a bounded fast/slow refinement and shared SiLU decoder produce twelve distribution parameters. The read is causal; the current observation also reaches the direct encoder/readout path.

Three causal stages (diagonal affine source scan, prefix evidence sum, matrix affine scan) admit linear sequence work and logarithmic scan span at fixed state dimension. This is an algebraic operation-count claim, not measured GPU speed or globally independent neural layers. Streaming updates have fixed-size state; batching trades additional intermediate storage for parallel sequence processing. No state-dependent matrix gates are passed off as input-only schedules.

## Forecast distribution

For each of six pollutants, latent log1p concentration L is Gaussian with mu = source anchor + 2*tanh(mean correction), sigma=.05+softplus(scale output), in FIT-standardized log units. The anchor is the actual current measurement if present, otherwise the normalized admitted source trace within supplied history. Physical Y=max(0,exp(L)-1). Its likelihood is a discrete atom at zero plus continuous density with the log1p Jacobian. Median and 5%/95% quantiles come from this SAME distribution; no independently inconsistent positive-delay head or posthoc interval inflation is present. Zero initial mean-correction weights start forecasts at the anchor. They remain trainable. All 97,235 parameters affect supervised outputs; no parameter padding.

## Identity, reuse and numerical contract

States carry caller-supplied station owner, model-instance token, parameter-version tuple and last observation hour. Foreign owner/model, changed weights and non-increasing clock reuse are rejected. This is identity validation, not discovered entity tracking or durable cross-process state serialization. Fixed core payload: 13+13+62+64=152 FP32 values=608 bytes/sample. Count/owner/last-hour add24 bytes; external source-age bookkeeping adds104 bytes. Python provenance, allocator overhead and batch intermediates are additional. Model/runtime memory is NOT608bytes total.

PNDS/SourceAlignedTrace.lean proves only exact-real source kernel bounds, no-observation identity and affine composition/action. Existing matrix bounds are inherited. Independent serial Python evaluates the entire graph using rank-one matrix updates rather than scan kernels and checks output/state/all-parameter gradients in FP32/FP64. Chunk/full, owner rejection and missing-numeric invariance are tested. Full neural-network IEEE Lean refinement remains open.

## Evidence boundary

UCI501 has real station identities, hourly measurements and naturally missing cells. It does not establish arrival times, verified late observations, action consequences, occlusion, source permissions or causal intervention outcomes. Model inputs assume observations available at their published measurement hour. Station meteorological measurements were publisher-matched from nearby weather sites; no independent meteorological agent is invented.

FIT2013–2014; REPORT2015 first opened after all five original models are Git-frozen; 2016–2017 numerical rows unavailable to this loader. Source is public, not a privately blinded prospective deployment. Primary cohort is conditional on genuine missing current PM2.5, age8..255hours, and observed+6h outcome. It is not an all-hour risk estimate. General daily queries are reported separately. Same trained weights score histories8/256; both lengths occur in training. This access probe is not a head-to-head architectural comparison. Classical refs use source-provided current/stale values only. No neural-superiority or full master-plan qualification follows from this pilot.
