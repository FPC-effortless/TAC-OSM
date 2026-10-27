# TAC-LANG-REPAIR-016/017 — mathematical repairs, source-verified risks and controlled next gates

Status: research-only; NOT a retrained or promoted model. Companion PNDS mathematical proofs on PR #51; TAC implementation PR #527. Existing 100,680 learned-parameter checkpoint family unchanged. No new seventh TAC family.

## 1. Observations versus hypothetical failures

**Proven structural issue:** The four Fourier periods are 7, 13, 29, 59; their least common multiple is 155,701. For every ordinal t, the original source address before the learned rotation at t+155701 equals the one at t in exact arithmetic. No deterministic downstream linear map, normalization or classifier of that same address can recover an era that is not represented.

**Proven local mathematical risk:** Original `F.normalize(x, eps=1e-6)` has derivative magnitude 1e6 near x=0 along a coordinate slice and a clamp-boundary kink.

**Observed on authentic FIT source, NOT an efficacy test:** Examined 10 frozen `TAC-LANG-GATE-003` checkpoints (5 SiLU / 5 raw-linear, misnamed tanh) on 16 fixed 128-token windows from the first four FIT chapters of the Vim 9.1 manual (2,048 token observations per checkpoint; 20,480 model-token occurrences). Before measurements, verified SHA256 of all 36 original Vim 9.1 chapters against the frozen PROTOCOL.json and the 10 checkpoints against RESULTS.json. No training, optimizer steps, or independent TEST access. Across the 10 checkpoint/fit-window combinations:

| Pre-normalization measurement | Minimum over 10 checkpoints |
|---|---:|
| Native 8D key norm | 0.52088 |
| Aligned 8D projected source key norm | 1.71583 |
| Source rotation minimum singular value | 0.74705 |
| Native / aligned norms below 0.01 | 0 / 0 |

This **does not support claiming a near-zero-key failure on the observed trained FIT inputs**. The risk may occur under distribution shift, quantization or later optimization and needs separate testing. Reproduce with `audit_fit_key_conditioning.py` and the original SOURCE+checkpoint archive (not yet publicly hosted in GitHub Actions).

## 2. Parameter-neutral smooth normalization candidate

```
d_tau(x) = sqrt(||x||_2² + tau²)
k_tau(x) = x/d_tau(x), tau = 0.01
```

Exact-real Jacobian:
```
J(x) = I/d_tau(x) - xx^T/d_tau(x)^3
lambda_tangent = 1/d_tau(x)
lambda_radial = tau²/d_tau(x)^3
||J(x)||_2 <= 1/tau = 100.
||k_tau(x)||_2 <= 1.
```

This is smooth for every x and eliminates the original 1e-6 branch kink. It bounds the worst local gain from 1e6 to 100 but **changes keys**, especially those with ||x|| comparable to tau; old trained weights cannot fairly establish its performance without re-training. Its nonunit length can weaken write strength. The Lean candidate proves a positive scalar denominator and a bound relative to tau; the full 8D operator norm is a paper derivation and independently autograd-tested, not yet a Lean theorem.

## 3. Learned source-rotation conditioning with no new parameters

Given original per-head orthogonal initial matrices Q (identity and 1-step rolled identity) and all 64 original trainable coordinates W per head, define

```
R(W) = Q + delta*tanh(W-Q),  delta=0.05, elementwise tanh.
```

All original trainable coordinates remain active; no dummy parameters. Since every entry of R-Q has absolute value < delta, its Frobenius norm is <= 8delta. Thus, for all v,

```
||R(W)v||_2 >= (1-8delta)||v||_2 = 0.6||v||_2.
```

Source Fourier phi(t) has exact norm 2. A bounded era tag with absolute size <=0.25 keeps ||phi_tagged|| >=1.75, so in **exact real arithmetic** ||R phi_tagged|| >=1.05. This prevents collapse of the aligned source address under any learned W. It restricts the learnable rotation geometry; training and real-data results may worsen. The candidate is separate from frozen original W.

## 4. The positional cycle fix and its hard limit

The candidate retains the original phi(t) in era 0 and adds a bounded first-coordinate tag after the first cycle:

```
P = 155701; q = floor(t/P)
phi_tagged(t)[0] = phi(t)[0] + 0.25*q/(q+1)
phi_tagged(t)[j] = phi(t)[j], j>=1
```

PNDS proves the exact integer pair `(q, t mod P)` reconstructs t and that q increments once per P. The exact-real tag strictly increases each era, separating **same-residue forced aliases before projection**. It does not prove all codes are mutually distant or preserve long-term retrieval. The tag eventually saturates in FP32; the test includes a large-era alias. The repaired full-rank rotation ensures linear projection does not collapse distinct *input vectors*, but normalized directions are not proven injective for arbitrary t.

**Unbounded exact historical identity needs an integer-tagged external store** (keyed by source owner and full ordinal) or a declared bounded historical horizon. The neural core can remain 408 floating state values, but external persistent records use additional storage and I/O. A fixed, finite-precision 408-value state cannot promise lossless arbitrary-length source replay. If external memory is used, every comparator must receive the same authorized source records and memory budget.

## 5. Bounded same-size fast state alternative

The original source keeps an 88-element cumulative sum and computes `context = sum/max(count,1)`. Alternative:

```
mu[0]=0
mu[n+1] = mu[n] + (basis[n]-mu[n])/(n+1)
```

PNDS already proves the exact-real mean step equivalence and unit-box preservation for bounded inputs. It requires the same 88 floating values, and the mean is bounded in [-1,1] with tanh basis. **FP32 is not universally better:** the constructed `late-small` sequence improves with the online mean; a `1,-0.99999` alternating sequence is more accurate with the original sum. Both counterexamples are retained in CI; do not promote based on arithmetic diagnostics alone.

## 6. Experimental program and gates

The recommended next low-compute staged comparison is: original frozen source (reference); smooth-only keys; bounded source rotation (with/without era code); online-mean ablation. Preregister the subset and run equal-parameter, equal training updates, identical FIT/DEV document partitions and equal evaluation conditions; no need to match CPU wall-time, but record compute and inference latency. Keep learned parameters within 95k–105k, comparator counts within 1%, real source as primary data, five seeds and replayable checkpoints. Do not use the constructed tests as claims of real-text performance.

Primary metrics on genuine time-ordered documents: next-character NLL; historical retrieval NLL and correct-source ordinal margin; prefix-state continuity; performance at 128/512/2048/8192 tokens; interference and relevant memory lesion; calibration; and measured incremental I/O cost for any external store. A >P position stress is only a diagnostic until actual texts and lawful source identities support it.

**Current research priority after FIT inspection:** prove/use source-identity stability and retrieval utility before tuning an unobserved near-zero key pathology. No architecture wins, causal memory benefits or investor-ready claims have been demonstrated in this repair phase.

## Context

Long-context position encodings in the literature also face extrapolation and positional-resolution trade-offs: ICLR 2026 *Frequency Bands in RoPE* and ACL 2024 *HiRoPE* offer relevant comparison concepts. These are references, not evidence that their solutions will work for TAC.
