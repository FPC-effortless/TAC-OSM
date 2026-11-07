# Mathematical engineering review: TAC-OSM CSL-125-R2 and PNDS

7 November 2026 · Internal source and numerical audit · Draft, no model promotion

## Assessment

The scalar recurrence has coherent exact-real semantics and useful conditional proofs. Its present implementation is **not numerically or mathematically qualified as a whole model**. This review reproduced the historical gradient failures, isolated failures beyond normalized-objective rounding, and found a separate, reproducible discrepancy between logical state size and retained tensor storage at the default 123,822,574-parameter configuration.

These findings justify targeted engineering work. They do not show that the architecture cannot learn, that a trained model is harmed by these cases, or that a different architecture is superior. No training, dataset scoring, optimizer update, GPU work, merge or production-source change occurred.

### Scope and provenance

- TAC source: `e0753b2c90ad7e21bffa2d66f009ab385d6d4645`, the inspected head of [audit PR #714](https://github.com/FPC-effortless/TAC-OSM/pull/714).
- Runtime graph: `csl125_r2.py`, `csl125_r1.py`, `csl125.py`, `csl001.py`; optional `csl125_r2_fast_scan.py`; original audit and NumPy reference.
- PNDS source: `0fa31e981681f511c601c5fd069432061061c509`, [PR #116](https://github.com/FPC-effortless/pnds-lean/pull/116), including the inherited R1 scalar module.
- Exact file hashes are in `SOURCE_HASHES.json`. Source snapshots were retrieved through GitHub and checked against the original operator pins. This was a selected-file source audit, not a complete repository checkout or full repository test run.
- Runtime: Python 3.12.14, PyTorch 2.14.1+cpu, NumPy 2.3.5; one Torch thread. See machine receipts for authoritative versions.
- Executed: original 84-row replay, 336 precision-isolation operator rows, 336 fixed-cotangent/interior operator rows, 184 independent reference-gradient comparisons, eight finite-difference directional sweeps, three small-model full/chunk gradient comparisons, contract counterexamples, and reduced/default-width storage probes. The 83 selected pytest cases passed.
- This report covers the CSL R2 lineage deeply. It does **not** certify all six TAC families, historical language architectures, Atlas, USEF-X, R3/R4 adapters, datasets or trained checkpoints. No result transfers automatically to those systems.

## Findings ordered by engineering consequence

| ID | Finding | Evidence and severity | Disposition |
|---|---|---|---|
| ME-R01 | Returned state retains substantially more storage than its tensor shapes indicate | Confirmed at default width; high priority for memory accounting and edge feasibility | Correct accounting now; separately test compact state ownership |
| ME-R02 | FP32 backward correspondence fails under cancellation and low occupancy | Original 11/84 failures reproduced; fixed-cotangent and interior checks still fail | Numerical qualification remains failed; isolate accumulation and parameter-chain effects |
| ME-R03 | Configuration accepts positive epsilon that becomes zero in FP32 | `1e-100` accepted, empty read produces NaN; default `1e-8` control is finite | Reject unusable machine-domain values; preserve state law |
| ME-R04 | Scalar proof preconditions are not enforced at carried-state ingress | Negative mass is accepted; closed gate does not isolate NaN write data | Define trusted-state invariant and validation boundary |
| ME-R05 | Exact-real retention identities do not fully describe FP32 behavior | Valid tiny update has beta exactly 1 in FP32 while eta remains positive | Add finite-precision retention contract; no demonstrated training harm |
| ME-R06 | Optional fast kernel has stricter rate domain than configurable model | Accepted `tau_hard_min=0.5` permits rates above the optional kernel's 0.5 maximum | Integration hazard, not an active default-kernel defect |
| ME-R07 | Formal coverage is scalar and conditional | Compiled R2 theorems cover hold/reduction/one-step bounds and integer arithmetic | Whole graph, source refinement, AD and IEEE proofs remain open |

### ME-R01: logical state is not retained allocation

The scan returns `result[:, -1]`, and numerator/mass are further views of that slice. The convolution tail is a view of the concatenated input history. KV caches are views of `all_k` and `all_v`. Small visible tensors therefore keep larger backing allocations alive.

Relevant original code: [scan end state and convolution tail](https://github.com/FPC-effortless/TAC-OSM/blob/e0753b2c90ad7e21bffa2d66f009ab385d6d4645/src/tac_osm/csl125_r2.py#L82), [KV slices](https://github.com/FPC-effortless/TAC-OSM/blob/e0753b2c90ad7e21bffa2d66f009ab385d6d4645/src/tac_osm/csl125.py#L161).

Measured under `torch.no_grad()`, batch 1, full 20-layer default model:

| Prefill tokens | Logical state bytes | Unique retained tensor-storage bytes | Compact-copy storage bytes | Next-token outputs |
|---:|---:|---:|---:|---|
| 64 | 1,171,456 | 46,391,296 | 1,171,456 | Bitwise identical |
| 128 | 1,558,528 | 50,716,672 | 1,558,528 | Bitwise identical |

Storage is deduplicated by backing-storage pointer, so numerator/mass aliases are not double counted. `state.detached()` retains the same storage sizes. These are tensor-storage measurements, **not** RSS, allocator reservations, peak RAM, or total inference memory. Weights are excluded. The existing R2 report's T64 value of 1,171,456 is correct as logical element count but understates retained backing storage by about 39.6 times in this probe.

The recurrent component retains at most the final scan chunk's allocation; it is not unbounded in total history for fixed chunk size. Convolution tails and KV views can retain allocations proportional to the size of the most recent call. A subsequent short decode call can replace these with smaller allocations. Thus this is not evidence of an indefinitely accumulating leak, but it invalidates using shape counts alone as the actual persistent-memory contract.

For default FP32 geometry, logical recurrent-plus-convolution state is `20*(4*8*256 + 2*768)*4 = 778,240` bytes. Full-window KV adds `4*2*3*127*64*4 = 780,288` bytes. Neither formula includes autograd graphs or allocator overhead.

The compact-copy probe establishes a plausible remediation, not a shipped fix. A repair must preserve attached training gradients, ownership, state version and all exits; do not insert `detach()` into training to save memory. Measure copy cost and peak allocation as well as retained storage. Test large prefill, token decoding and changing chunk partitions.

### ME-R02: failures survive objective isolation

The unchanged historical audit reproduced 11 failed rows out of 84, including the same near-empty carry and stationary rate categories. All historical tolerances remain unchanged: gradient `atol=5e-6`, `rtol=5e-5`; forward `atol=rtol=1e-5`.

The new precision matrix has 56 cases per operator per objective mode: seven fixtures, two epsilon values, four chunk sizes. A row fails if any of the four argument-gradient tensors misses tolerance.

| New diagnostic objective | Stable scan failures / 56 | Optional scan failures / 56 |
|---|---:|---:|
| FP32 objective, coefficients formed once then cast | 11 | 10 |
| FP64 normalized objective over original FP32 scan outputs | 10 | 11 |
| Exactly representable dyadic coefficients | 5 | 6 |

These rows do not replace the original audit. The new coefficient construction and scalar-loss scale differ; especially, the dyadic objective uses a power-of-two normalization rather than a mean over 260 values. Fewer failures under a differently scaled loss are not an improvement in the kernel or a gate pass.

FP64 objective evaluation still sends gradients through a cast into an FP32 scan backward path. It does not make the whole backward computation FP64. To remove this ambiguity, the follow-up froze one cotangent tensor: compute it from the FP64 reference, round it **once** to FP32, then use those identical numerical values in both precisions. This compares vector-Jacobian products with the same upstream derivative.

| Fixed-cotangent / interior follow-up | Stable failures / 56 | Optional failures / 56 |
|---|---:|---:|
| Original inputs, identical frozen cotangent | 11 | 11 |
| Interiorized inputs, identical frozen cotangent | 8 | 10 |
| Interiorized inputs, normalized objective | 11 | 11 |

Interiorization halves write and carry numerators and clamps gates to `[1e-6, 1-1e-6]`; masses, addresses and rates stay fixed. These inputs satisfy the operator's interior bounds. This is not a proof that all such tensors arise jointly from a particular trained model history.

**Supported conclusion:** discrepancies are not solely caused by constructing the normalized loss in different precisions, and do not require exact value/gate saturation. The FP32 recurrence-Jacobian contraction remains implicated. This does not identify one unique kernel instruction, establish a wrong exact-real recurrence, or demonstrate harmful optimizer updates.

### Independent reference validation

An independently written NumPy analytical reverse recurrence matched the FP64 Torch AD reference for all 42 case/epsilon/objective combinations and all four input arguments. Four additional near-empty/stationary comparisons used Decimal precision 80 for the complete scalar-time reverse calculation. All 184 argument-tensor comparisons passed `atol=1e-8, rtol=1e-7`.

Eight normalized-objective directional finite-difference checks used steps `1e-3` through `1e-7`; all step results are retained rather than selecting a favorable step silently. The minimum absolute discrepancy per sweep ranged from approximately `1.1e-14` to `1.1e-11`. These finite differences used a well-conditioned interior fixture. They do not independently validate every coordinate in every ill-conditioned case; Decimal checks address selected difficult cases separately.

An additional one-step hand-derived test distinguishes the correct rate/gate chain factors from a missing-factor mutation. This reduces the risk of two references agreeing through the same coding error; it is not formal verification.

### Whole-graph positive controls and their limits

The original inherited graph was instantiated in a 6,827-parameter geometry: two blocks, local GQA in block 2, width 16, state width 8, two roles, vocabulary 32. Seeds 11, 22 and 33 compared T13 full execution against chunks `[3,4,6]`, with attached carried state and unchanged stream identity.

All outputs passed forward tolerance; all 47 parameter-gradient tensors were present and passed the frozen gradient tolerance in each seed. No gradient tensors were omitted. The scalar diagnostic included every emitted head so otherwise-disconnected task heads were not silently ignored. Maximum scaled gradient errors were below 0.0032, where 1 is the threshold.

This supports ordinary small-graph chunk consistency. It does not establish FP64 whole-graph derivative equivalence, training-loss correctness at 125M scale, long-history gradient health or learned usefulness. The default-width tests in this review were storage and next-token-copy checks, not a full-size backward audit.

### ME-R03–R06: boundary contracts

1. **Epsilon representability.** `CSL125R1Config` checks positive finite epsilon and an upper bound, but no FP32 representability floor. `1e-100` passes configuration validation and becomes zero in the implemented FP32 denominator. Empty numerator/mass then yields `0/0`. The default `1e-8` control remains finite. This is an edge-configuration defect, not evidence that the default routinely produces NaNs.
2. **State validity.** Public carried-state handling checks shapes and stream IDs, not finite values, nonnegative mass or `|N|<=D`. A constructed state with negative mass was accepted and produced finite logits. Finite logits do not establish a valid state. If states are exclusively trusted internal outputs, validate at restoration/import boundaries rather than forcing a synchronizing check in every token hot path.
3. **Closed gates and NaNs.** With `g=0`, the code still evaluates `eta*z`. If `z` contains NaN, `0*NaN` contaminates the state. The exact-real no-write theorem is valid for real inputs and says nothing about NaNs. Document finite-input assumptions and decide whether missing rows require structural masking before learned arithmetic. Do not represent multiplication by zero as a missing-data operator.
4. **Machine retention.** For the valid tiny-update fixture, `r*g = 3.0517577939530305e-13`; FP32 `beta=1` and `eta>0`. Therefore the separately rounded coefficients do not implement the exact-real complementary pair as real numbers. This demonstrates loss of the exact contraction identity, not an observed unbounded state or a measured failure on long real sequences.
5. **Optional-kernel domain.** Configuration permits `tau_hard_min=0.5`, whose rate supremum is 2. The optional cumprod kernel admits only rates at most 0.5 and rejects larger rates. The default model has `tau_hard_min=2`, and does not call that optional kernel. The rejection is protective; a future dispatcher must encode the domain explicitly.

## Mathematical derivation and proof obligations

The derivations below are analytical review results, not newly compiled Lean theorems.

### Typed causal graph

At each layer, input `x[B,T,d]` is RMS-normalized. The local branch uses a causal depthwise convolution and a retained normalized-input tail. The memory branch maps normalized features to queries `q[B,T,E]`, softmax role weights `p[B,T,R]`, bounded values `v[B,T,E]`, and sigmoid write gates `g[B,T,R]`. Fast and slow rates have shape `[R,E]`. Each branch stores numerator and mass `[B,R,E]`.

The pair write is `(p*v,p)[B,T,R,E,2]`. Scans are inclusive: the token's own write participates in its current read. That is causal for predicting the next token; it is not a read-before-write event semantics. Role weights also mix the normalized slot reads. Learned read projections and a learned read gate feed the residual stream, followed by a gated FFN residual. Selected layers additionally run bounded-window GQA. Final normalized features feed tied token logits and task heads.

The pipeline has no built-in document-reset mask in the recurrence. A target loss mask does not erase prior state. Stream IDs protect accidental row/stream mismatches, and changing active depth requires reset/replay. Dataset packing, restoration and per-event timing still require their own source-level adapters.

### Exact-real recurrence and invariants

For one role/channel, write `z_t=(p_t v_t,p_t)`, `u_t=r g_t`, `beta_t=exp(-u_t)` and `eta_t=1-beta_t`:

\[
S_t=\beta_tS_{t-1}+\eta_t z_t.
\]

Under `r>0`, `0<=g<=1`, `p>=0`, `|v|<=1`, and an initial state in the cone `D>=0, |N|<=D`, each update preserves that cone. If additionally `D_0<=1` and `p_t<=1`, exact-real mass stays at most 1. This follows by induction from convexity; the current R2 Lean module proves the paired one-step inequalities, not this complete source-level induction and all input obligations.

The unrolled solution has old-state weight

\[
w_0=\exp\!\left(-r\sum_{j=1}^t g_j\right),\qquad
w_i=(1-e^{-r g_i})\exp\!\left(-r\sum_{j=i+1}^t g_j\right).
\]

All weights are nonnegative and `w_0+sum_i w_i=1`. For fixed externally supplied gates, differences between two states contract by `w_0`. There is no uniform strict contraction when gates may remain zero. When gates depend on hidden states, a full-network Jacobian also includes their derivatives; the fixed-gate contraction cannot be applied unchanged.

Retention runs on cumulative gate exposure, not simply elapsed tokens. With constant gate `g>0`, old-state half-life is `tau*log(2)/g`. Closed gates hold information but also prevent new admission. Bounded state, long survival and successful retrieval are separate properties.

### Read conditioning

For `y=N/(D+epsilon)`, valid state and positive epsilon imply `|y|<=1`, but

\[
\partial_N y=\frac{1}{D+\epsilon},\qquad
\partial_D y=-\frac{N}{(D+\epsilon)^2}.
\]

The coordinate bounds are `|partial_N y|<=1/epsilon` and, using `|N|<=D`, `|partial_D y|<=1/(4 epsilon)`. At the default epsilon, these permit sensitivities of order 100 million and 25 million. These are worst-case coordinate bounds, not observed full-model gradient magnitudes.

For finite perturbations,

\[
\Delta y=\frac{(D+\epsilon)\Delta N-N\Delta D}
{(D+\epsilon)(D+\epsilon+\Delta D)}.
\]

Correlated perturbations can cancel. Treating numerator and mass as independent coordinates gives a deliberately broader test than perturbations reachable through their shared neural parameters. Conversely, small forward error does not bound derivative error without a conditioning argument.

With stationary `N=D`, the read equals `D/(D+epsilon)`. Differentiation along this tied path gives `epsilon*D'/(D+epsilon)^2`, while an unfused chain computes the difference of two larger terms. This explains a cancellation mechanism consistent with the rate failures. It is not a proof that this is the only contribution in every failed row.

### Backward recurrence

Let `lambda_t` be the accumulated state adjoint, including the local objective derivative at time t. Then the exact-real reverse equations are

\[
\nabla_{z_t}L=\eta_t\lambda_t,\qquad
\nabla_{u_t}L=\langle\lambda_t,\beta_t(z_t-S_{t-1})\rangle,
\qquad \lambda_{t-1}\mathrel{+}=\beta_t\lambda_t.
\]

Rate gradients sum `g_t * grad_u` over time and batch; gate gradients sum `r * grad_u` over channels. The independent reference implements this recurrence directly. A future custom backward could exploit these expressions, but reassociation, accumulation precision, higher derivatives and compiler behavior would need a separate audit.

For learned gate logit `a`, `dg/da=g(1-g)`. For rate logit `b`, with span `log(tau_max/tau_min)`, `dr/db=-span*r*sigmoid(b)*(1-sigmoid(b))`. Thus a failed raw rate or carry coordinate does not directly quantify its effect on trained parameter updates. Saturation can suppress gradients even when the operator derivative is large. No trained-state distribution or optimizer-effect study was performed here.

### Parallelism, capacity and complete-network stability

Affine maps compose associatively over the reals: `(A2,B2) after (A1,B1)=(A2*A1,B2+A2*B1)`. The stable implementation uses doubling within chunks and serial carries across chunks. With chunk length C, it has O(T log C) work and O(log C) depth per chunk; it is not a globally logarithmic-depth implementation across arbitrary T. Floating-point reassociation requires its own error contract.

The optional formula divides by prefix products. The lower bound `exp(-32)` excludes one underflow mechanism under its declared domain; it does not prove well-conditioned signed sums or gradients. Nonzero denominators are not a complete numerical-stability theorem.

Normalized weighted averages compress histories. In the constant-address/equal-rate-and-gate case, address scale cancels from the unregularized ratio; separate role labels alone do not create separate stored contents. With epsilon, cancellation is approximate and occupancy-dependent. This limitation was already recognized in the R2 mathematical note and is not presented as a new discovery.

The bounded memory read does not bound the learned read projections, residual gains, FFN, attention output projection or the full Jacobian. The current source provides no complete-network contraction theorem. A useful next theorem should name a norm, parameter constraints, input domain and stability notion; otherwise a scalar EMA result cannot certify the residual architecture.

## PNDS coverage audit

I retrieved the historical [build run 42044323931](https://github.com/FPC-effortless/pnds-lean/actions/runs/42044323931) and job log. The log explicitly prints the R2 statements and their axioms. The five real-arithmetic theorems use `propext`, `Classical.choice`, and `Quot.sound`; the integer ceiling theorem uses no axioms. The build completed successfully. This verifies the recorded compilation evidence; Lean was not installed or rerun locally in this review.

| Formal item | Established scope | Still missing |
|---|---|---|
| Zero-gate numerator/mass hold | Exact real arithmetic | Finite inputs, IEEE/signed zero/NaN semantics, neural gate attainability |
| Full-gate R1 reduction | Exact real formula | Numerical equivalence across actual implementations |
| One-step upper/lower bounds | Supplied beta, address, value and prior-state hypotheses | Derive hypotheses from the typed graph; inductive full stream invariant |
| Parameter ceiling | `123822574 <= 125000000` | Formal source-to-count correspondence; a numeric proposition alone is not model extraction |
| Normalized read | Analyzed here in prose/Python | Compiled conditioning, perturbation and gradient theorems |
| Scan | Exact algebra reviewed here | Prefix invariant, floating-point reassociation and backward correspondence |
| Full model | Source traced and selected tests executed | Complete Lean semantics, masking/time/state rules, full source refinement |

The canonical variant manifest remains a partial historical index. The R2 sidecar explicitly declares noncanonical draft status. Do not silently insert a new certification label or claim these results qualify the other five consolidated families.

## Review integrity and reproduction

The initial diagnostic mislabeled coefficients as dyadic while still dividing by 260. That first result and script are preserved as `RESULTS_INITIAL.json` and `script_initial.py`; the label is withdrawn. The preregistered amendment changed the dyadic normalization to a power of two and retained all other outcomes. The exact executed phase-2 script is preserved as `script_phase2.py` and its SHA matches `RESULTS_PHASE2.json`. The final runner was also replayed after making storage geometry configurable; its receipt is `RESULTS.json`. Later storage and VJP additions have separate pre-execution registrations. No tolerance was relaxed.

The replay bundle includes source snapshots, registrations, scripts, tests and raw JSON evidence. Use a fresh extraction directory. Install `torch==2.14.1+cpu`, NumPy and pytest; see `REPLAY.md`. Scripts refuse to overwrite their result files. Original model code remains byte-identical. The archive manifest verifies bytes, not independent scientific provenance. Local tests are internal engineering evidence, not external qualification or new hosted CI.

## Repair and verification sequence

1. **State ownership/accounting:** implement a separate compact-state change; test gradient-preserving copies, every state component, all exits, long prefill and token decode. Acceptance: bounded retained storage consistent with the declared contract, unchanged outputs/gradients within original tolerances, measured copy overhead. Keep logical bytes, retained storage, allocator memory and RSS separate.
2. **Numerical backward:** preserve all 11 original failures. Compare reference adjoint recurrence, compensated/higher-precision accumulations and source parameter-chain errors under one fixed objective. Acceptance: frozen case coverage and tolerances; no substitution of a scaled objective for the original gate. Establish effects on realistic trained-state distributions before claiming a training benefit.
3. **Input/state contract:** reject nonrepresentable epsilon and explicitly handle restoration validity. Define whether nonfinite writes are errors or masked observations. Acceptance: invalid states fail at the chosen boundary, finite trusted paths remain unchanged, no accidental detachment or hot-path synchronization regression.
4. **PNDS correspondence:** formalize the cone induction, normalized-read bounds and backward recurrence with explicit hypotheses; then connect them to source shapes, gates, rates and ownership. Compile and inspect axioms; match executable references to exact source commits in both repositories. Numerical tests do not replace this work.
5. **Capability work:** only after relevant engineering gates, use authentic registered data, matched controls and checkpoint replay to test learned admission/retrieval and actual resource-quality tradeoffs. None of this review's synthetic fixtures is primary capability evidence.

The immediate engineering priorities are the confirmed storage discrepancy and unresolved backward precision. A global architecture rewrite, tolerance relaxation or new large training run is not justified by this audit.
