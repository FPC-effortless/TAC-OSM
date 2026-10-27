# PAR-MEM-010 — restoring earlier TAC memory mechanisms into the parallel family

**Status: IMPLEMENTED / ORIGINAL REAL SOURCE NO-FIT OPERATOR PREFLIGHT; NO trained
historical retrieval success; no independent model qualification or merge.**

This research version directly follows [PAR-MEM-009 master-plan and implementation
reconciliation](PAR-MEM-009-PRIOR-SOLUTIONS-REVIEW.md), and explicitly reuses
source-documented older mechanisms rather than starting another unsupported
memory redesign. The original evidence and trained checkpoints remain unchanged.

## Actual integrated code (new optional candidate)

- [Source-aligned parallel TAC PyTorch implementation](../../src/tac_osm/variants/source_aligned_parallel_world.py):
  retained original `TimeAwareParallelResidualWorld` (96,639 learned
  parameters); added two sidecar 16-value × 8-key source memory heads,
  input-only *source-known* ordinal Fourier keys, a shared learned key
  transformation for both WRITE/READ, separate per-head sigmoid retention and
  strength maps, differentiated soft writing, per-source owner/chronological
  event guards, a legal read-only historical-content head, and an
  **exact-zero-initial** corrective predictive path.
- [Existing SIX-family research factory](../../src/tac_osm/variants/parallel_six.py)
  opt-in `instantiate_untrained_source_aligned_core`: no new seventh family,
  no qualified task heads, no replacement of original trained base models.
- [No-fit unit operator tests](../../tests/test_source_aligned_parallel_world.py):
  same source address despite changing current physical event content,
  no-current/future leakage, read-only state, source-owner rejection, future
  timestamp and nonmonotonic index rejection, exact no-source-write identity,
  batch/stream equivalence of 2×16×8 aligned content memory and predictions,
  neural graph differentiability and six-family registry behavior.
- [Genuine published physical source no-fit preflight](../../scripts/parmem010_real_source_nofit.py):
  original SHA-pinned UCI Beijing physical sensor ZIP, verified 12 independent
  real station identities and 16 physical observations each from genuine
  March 2014 FIT, source-provided original publisher hour ordinal,
  true observed PM2.5 content only at its normal observation time.
  No future source outcomes or historical reconstruction **scored**.
- [Formal companion draft PNDS PR #49](https://github.com/FPC-effortless/pnds-lean/pull/49):
  source-ordinal shared-address equality, owner and future-query denial,
  local corrective row write and exact no-write and read-only operator
  identities in Lean 4; independently executed scalar Python reference.
  This is operator-only mathematical engineering, NOT full PyTorch
  finite precision, reverse-mode AD or probabilistic neural refinement.

### Precisely counted state and learned parameters

| Memory or resource | Physical measured quantity |
|---|---:|
| Original time-aware running context and associative matrix | 154 FP32 values |
| New retained source-known 2×16×8 sidecar (no event KV history) | 256 FP32 values |
| **Entire numeric streaming state** | **410 FP32 values = 1,640 bytes** |
| Additional logical and provenance metadata | Native prior-write Boolean + count; 64-bit source owner + 64-bit last source event ordinal; Torch/Python/runtime overhead additional |
| Actual live and deployed learned model weights | **98,326 parameters**, including the previous 96,639 |
| Trained steps on new architecture | **0** |
| Six current research families | **6, unchanged** |

This is NOT equivalent to the old LATFUSE 1,696-float full source-slot
state or ME100's 152-float plus additional bookkeeping, and no inference
latency or device memory is measured here. The new 410-float version does
not implement LATFUSE's 16 source slots, separate persistent fast80/slow80
states, trained event utility, CED refinement or a full end-to-end
probability law. It is an explicitly narrower opt-in adaptation of
historical source-aware address and evidence concepts.

### Real-source operator outcome, not prediction skill

The ORIGINAL [first source and software validation CI](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40517395840)
passed 10 new source-aligned constructed operator tests plus 9 original
time-aware regressions (19 focused tests total). Publisher original ZIP
SHA256 `d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`.

On 12 real observing stations ×16 original March 2014 legitimate
measurements each = **192 source-observed real admissions**, the absolute
maximal batch-vs-stream discrepancies were:
- all predicted normalized output coordinates: **5.960464477539063e-08**
- all new source-aligned state coordinates: **4.470348358154297e-08**

All timestamps were 2014 March first publisher hour indices 8760–8775,
with no source-inferred identity, no 2015–2017 numerical observations,
no outcome labels scored and **zero optimizer updates**.
This is *numerical implementation consistency on genuine inputs*, not
historical content retrieval accuracy, improved forecasting, calibrated
uncertainty, physical action consequences or independent model
generalization.

The registry-adapter tests added after the original run must also pass
their own updated CI to count as verified. Full Lean/PNDS CI for PR #49
must likewise pass separately.

### Lawful source-memory semantics

For source event ordinal `i`, let deterministic known-address features
`phi_i` be normalized Fourier features of `i mod {7,13,29,59}`.
Each head has its own shared learned map:
`k_h(i)=normalize(phi_i A_h)`. Inference can request an old ordinal `i`
and recompute `k_h(i)` **without storing the old original key or using
the old PM2.5 target as query information**.

With current legitimately observed content embedding `e_t`,
`v_t=tanh(W_v e_t)`, `alpha_h=sigmoid(W_alpha e_t)`,
`beta_h=sigmoid(W_beta e_t)*softmax(W_route e_t)_h`, the
per-head value×address write is:

`M_h,t = alpha_h M_h,t-1 + beta_h (v_t-alpha_h M_h,t-1 k_h(i_t)) k_h(i_t)^T`

when source-admitted; `M_h,t=M_h,t-1` EXACTLY on rejected writes.
Two 16D content rows split into 8D blocks yield four original
input-only 8x8 associative affine prefix scans, each sharing the
same chronological `alpha,beta,k_h` for that head. All batch
predictions use the **previous** memory prefix; the current write
cannot alter the same event's read.

Source owner + monotonically increasing event ordinal is REQUIRED,
but it is only source-name and order validation. The model **cannot
prove a particular requested ordinal had an observed PM2.5 label**:
data adapters / Woyengi source authority must do this. It does not
solve older event write-starvation, true old-value interference,
source-specific unseen IDs or temporal prediction optimization just
by making key-address geometry broader.

### Registered next trained capability gate (NOT yet authorized/fitted)

1. Source-label provenance: real distinct entity IDs/observed-at
   timestamps, target legal to observe at historic write time and
   expected time of later query. Any real previous source data has
   trained/exposed splits documented. The repeatedly examined 2014
   samples cannot be called an independent TEST.
2. Full architecture preflight: 95,000–105,000 actual live parameters
   per arm; <=1% between neural comparators, same measured real
   training data/splits and complete forward+autograd+gradient clip+
   optimizer FLOP budget. CPU runtime itself not a parity obligation.
   If unequal state bytes, report explicitly rather than silently
   claiming memory-size parity.
3. Source-time joint objectives: actual historic content losses at
   >=16/64/128 observed-event lags + real future physical conditional
   prediction loss, with old content never passed to legal query.
   Do not infer retrieval ability from key effective rank or
   cosine alone; include zero matrix, meaningfully retained fast/slow,
   source-time-only, source recent-value/seasonal prior and capable
   learned GRU/SSM controls.
4. A real untouched cohort (distinct site/time) must remain unopened
   through training, model selection and original five-seed checkpoint
   freezing. Preserve full model+optimizer/RNG/data order/stream
   state in Git and independent clean-checkout actual-source replay.
5. PNDS mathematical work must bind exact matrix scan to Torch's
   implemented four 8x8 blocks, owner and history-query semantics,
   finite-precision behavior and full derivative graph before claiming
   engineering completion; then separately quantify model utility.

**Conclusion:** this stage *restored source-aligned memory mechanics* and
made it operationally accessible through the six-family registry with
scientific and resource constraints intact. **Whether it retains any
specific event information better than original simpler models is OPEN.**
