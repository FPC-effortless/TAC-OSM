# ME105: complete input-to-output equations and refinement ledger

This specifies the **actual Temporal candidate**, not all six families. Source
will be pinned before any optimization. Python executable formulation:
`me105_graph_reference.serial` → `me102_graph_reference.serial` →
`me101_graph_reference.serial` → `me100_graph_reference.serial`. The reference
uses actual learned tensors but independent chronological equations, not the
production prefix kernels or any model `.forward`. Lean counterpart:
`PNDS/ProtectedSensor.lean`. Exact-real definitions/proofs, numerical source
agreement and empirical capability are distinct evidence levels.

## 1. Typed observations, state, information and validity

Batch B, nonempty history T. Values X, mask O and normalized ages A have shape
[B,T,13]; clock K [B,T,6]; station e [B] int64 in0..11; strictly increasing
publisher hours H [B,T] int64. X is finite training-standardized data clipped to
[-8,8]. A in[0,1] means min(source_age_hours,256) encoded by log1p(age)/log257.
Six pollutants use standardized log1p targets; seven contextual channels include
the published wind-tag sin/cos. Station embedding is [12,4]. No intervention,
action consequences, geography or missing labels are invented.

A stream state belongs to one model instance, parameter-version tuple and station:
S,C in R^13; E in R^62; count n integer; M in R^(8x8); last_hour, owner.
Updates reject foreign/stale/cross-entity/out-of-order state. Payload float32 bytes
=4*(13+13+62+64)=608; count/owner/hour int64 add24 per sample; metadata and six-channel
model buffers are separate. The manifest736-byte convention includes inherited
buffers; it is not a measured whole-runtime RAM footprint. Initial S=C=E=M=0,n=0.

## 2. Encoder, causal source admission and evidence prefixes

All operations below vectorize over batch and channels. Let safe_t=O_t*X_t,
x_t=tanh(W_enc[safe_t,O_t,A_t,K_t,embedding(e)]+b_enc) in R^48.
beta_t=O_t*sigmoid(W_adm*x_t+b_adm) in R^13. Source updates are

    S_t=(1-beta_t)*S_(t-1)+beta_t*safe_t/8
    C_t=(1-beta_t)*C_(t-1)+beta_t

Thus an unavailable channel has exactly identity S,C update, but other evidence
and memory may still update from observed channels. Define U_t=any(O_t), zero8,
and basis_t=last62(tanh(first88(fast_map[x_t,zero88,zero8]))).
E_t=E_(t-1)+U_t*basis_t, n_t=n_(t-1)+U_t.
The pre-event context c_t=[S_(t-1)/max(C_(t-1),1e-6), C_(t-1),
E_(t-1)/max(n_(t-1),1)] in R^88. It never reads future observations. The source
and evidence prefix schedules are computed from inputs alone, before M.

## 3. Fast/slow proposals and native associative writes

For pair-output fast_map: R^144→R^176 and slow_map: R^184→R^176,

    (p,g)=fast_map[x_t,c_t,zero8]
    f_pre=(1-sigmoid(g))*c_t+sigmoid(g)*tanh(p)
    (d,j)=slow_map[f_pre,c_t,zero8]
    s_pre=(1-.25sigmoid(j))*c_t+.25sigmoid(j)*tanh(c_t+tanh(d))

Both proposals have88 coordinates; the current source is NOT the old32D+32D core.
Normalized k_t=normalize(address(x_t),eps1e-6), v_t=normalize(content(x_t),eps1e-6)
in R^8. Inner gate logits(a,b)=gate_linear[x_t,f_pre,s_pre]. Controlled gates:

    a0=min(1-.02*(1-sigmoid(a)),1-1e-5)
    b0=.25*sigmoid(b)
    alpha=a0*(1-b0/4), beta=b0

The implementation logit/sigmoid wrapper realizes these real equations; rounding
is separately audited. For U_t true,

    M_t=alpha*M_(t-1)+beta*(v_t-alpha*M_(t-1)*k_t)*k_t^T
       =M_(t-1)*P_t+Q_t
    P_t=alpha*(I-beta*k_t*k_t^T), Q_t=beta*v_t*k_t^T.

Otherwise P=I,Q=0. Keys have norm <=1 (not guaranteed exactly1 at the epsilon
boundary). Write schedules depend on input/source/evidence prefixes, not previous
M. Consequently chronological affine composition
(Pa,Qa) then(Pb,Qb)=(Pa*Pb,Qa*Pb+Qb) is legal for parallel scan. Production uses
work-efficient O(T) fixed-dimension work/O(logT) span prefix and O(T) training
activations. Streaming state is constant in T. No general hardware-speed claim.

## 4. Distinct causal read paths and original forecast

At last observed index T, the original decoder reads **M_(T-1)**:
q=normalize(k_T+query[x_T,f_pre_T,s_pre_T]); r=M_(T-1)*q.
(fproposal,g)=fast_map[x_T,f_pre_T,r];
f=(1-sigmoid(g))*f_pre_T+sigmoid(g)*tanh(fproposal).
(d,j)=slow_map[f,s_pre_T,r];
s=(1-.25sigmoid(j))*s_pre_T+.25sigmoid(j)*tanh(s_pre_T+tanh(d)).
Raw=(Linear132→12∘SiLU∘Linear232→132)[f,s,r,x_T]
    +(Linear32→12∘SiLU∘Linear8→32)[r].
Split Raw=(delta,noise), each6. Anchor a is observed safe_T if available, otherwise
8S_T/max(C_T,1e-6). Base location m0=a+2tanh(delta), future scale s0=.05+softplus(noise).

A second **inclusive M_T** read uses the same write address of current masked
event: r_b=M_T*normalize(address(x_T)). Re-encoding current x_T in Python produces
the same result; the 102-vector phi=[8S_T/max(C_T,1e-6),C_T,E_T/max(n_T,1),A_T[:6],r_b]
contains13+13+62+6+8 coordinates. This query cannot feed back into its own scan.

## 5. Current belief, reconstruction and protected distribution

d_c=expm1(A_T,c*log257); missing I_c=1-O_T,c.
With positive softplus coefficients u[3,6],
V_lat=I*(u0+u1*d/24)*(1+u2*(1-clamp(C_T[:6],0,1))).
Response J=1+.25tanh(state_response). Current-head output(c_raw,v_raw)=Linear102→12(phi):
m_current=a+2tanh(c_raw), V_current=.0025+softplus(v_raw)^2+V_lat.
The inherited conditional affine correction is

    m_parent=m0+J*(m_current-a)*I
    V_model=s0^2+J^2*V_current*I
    F=I*FIT_floor_rate*d/24
    s_parent=sqrt(max(V_model,F)).

This J is the response of a declared conditional affine emission, **not** the
Jacobian of the complete network, nor an identified latent posterior.
New zero-initialized heads: width Linear102→12 (1,236params), median Linear102→6
(618params). Total100,349 trainable/deployed,33tensors. Define

    m=m_parent+.5tanh(median_head(phi))
    g_left/right=exp(log4*tanh(width_head(phi)_left/right))
    L=max(sqrt(F), .8*s_parent*I)
    s_left/right=max(s_parent*g_left/right,L).

Hence side variance>=F and missing-channel scale>=.8*s_parent. Both scales remain
positive, multiplier lies[1/4,4], median correction lies[-.5,.5] in standardizedlog
coordinates. Zero heads reproduce every parent quantile (sqrt(F)<=s_parent).
The .8 relative bound is a prospective exposed-DEV-motivated design decision,
not a calibrated90% guarantee; learned parent scale can still move. Missingness
cannot shrink below the protected lower bound at fixed other inputs, but no
monotone complete-model uncertainty-in-age theorem is claimed. Safe sqrt selects
argument1 before rooting zero F, then returns0 there, avoiding inactive-branch
infinite derivatives in autograd. Non-smooth boundaries use PyTorch subgradients.

## 6. Probability law, raw outcomes and objectives

Let epsilon standardnormal. Z=m+s_left*epsilon if epsilon<0, else m+s_right*epsilon.
Each half has probability1/2; CDF(z)=Phi((z-m)/s_side), continuous at m with possible
PDF jump. This is **not** the standard continuous-density two-piece normal.
Y=max(0,exp(target_mean+target_std*Z)-1). For y>0 transformed z=(log1p(y)-mean)/std:

    NLL=.5*((z-m)/s_side)^2+log(s_side)+.5log(2pi)+log(std)+log1p(y).

At y=0 the NLL is -logPhi((threshold-m)/s_at_threshold), threshold=-mean/std;
side is chosen at threshold, not always left. Quantiles are the same transform of
m+s_side*Phi^-1(p), with the inherited fixed z95 for the default90% interval.
Six conditional marginals at a six-hour horizon are NOT a joint multihorizon
trajectory or causal action model. Marginal log scores do not identify covariance.

Train loss=NLL+c_point*MAE(rawmedian/FITrawstd)+.1*loglocationMSE
+.3*maskedcurrentGaussianNLL+c_interval*raw90intervalscore/FITrawstd.
The interval score is width+20*lower_miss+20*upper_miss. Each loss masks authentic
labels only. Artificial gaps hide recorded training evidence, never invent labels.
For a positive target, dNLL/dm=(m-z)/s_side²,
dNLL/ds=1/s_side-(z-m)²/s_side³ away from side boundaries. Point loss has no direct
width-head gradient because z50=0; it DOES train inherited location and the new
median head (unless zero-censored). Variance and point losses share backbone
paths, so nonzero gradients alone do not imply useful memory or compatible tasks.
FIT FLOW_TRACE records actual block norms/cosines; conflict is batch-local evidence.

## 7. Executable engineering and remaining proof obligations

Thirty fitted FP32/64 checks must compare actual source vs independent serial
reference: outputs, S/C/E/M, features, both reads, source floor, median correction,
input/parameter gradients, zero atom/NLL, and32+32chunk carryover. Immutable limits:
FP32rtol2e-5/atol2e-6; FP64rtol1e-9/atol1e-10. Raw-unit absolute discrepancies must
be reported separately; these tolerances do not imply fixed absolute raw error.
Unit tests exercise protected-floor saturation and zero-age derivative boundaries.

Lean typed state and graph definitions plus explicit algebraic lemmas are limited
exact-real mathematics. Full PyTorch-network Lean refinement, nonlinear gradient
semantics, Gaussian measure/censoring normalization, IEEE rounding/error theorem,
statistical coverage, useful memory, cross-task transfer and safe decisions remain
OPEN. Do not promote a typed definition or function assumption into a theorem of
trained behavior. All six families remain unpromoted; this trains Temporal only.
