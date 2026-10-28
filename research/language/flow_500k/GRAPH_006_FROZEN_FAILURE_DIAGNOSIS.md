# TAC-LANG-GRAPH-006 — frozen trained-model failure diagnosis

**2026-10-28.** Status: diagnostic evidence on previously exposed real Vim 9.1 DEV; **FAIL**, no new fitting, no protected TEST. Does not replace PR #528's original observations or PR #529's prior trained-result report. Exact v2 source Git blob `d14c8a0cf562db15db5dcb555460126aba8a4666`; history adapter `8815cb8f11a1ab898dc6ac3e2aa8f3ebc0e09cf5`. 25 archived checkpoints were used read-only; checkpoint binaries are **not yet repo-backed**. Six authentic DEV source documents were independently SHA-checked against the original 36-document protocol.

## Main causal mechanism: the old-state signal is destroyed

In each aligned memory head,
`M_(t+1)=M_t P_t+Q_t`, `P_t=alpha_t(I-erase_t*k_t*k_t^T)`,
`Q_t=beta_t*v_t*k_t^T`, `beta_t=(1-alpha_t)*sigmoid(write_logit_t)`.

The v2 coupled write is a valid *conditional boundedness* repair but creates a direct optimization tradeoff: stronger immediate writes require reducing retention. For the **same previously exposed authentic 128-character DEV windows** and five seeds:

| Checkpoint family | Mean aligned alpha ↑ | Actual beta |
|---|---:|---:|
| Original 500K | 0.56136 | 0.83116 |
| Corrected graph-on v2 | **0.17510** | 0.69527 |
| Corrected v2 history-supervised | **0.13744** | 0.78401 |

Next-character optimization has learned to make most aligned old state disappear quickly. The original was also inadequate for actual 192-token recall. To survive a lag, the old-state Frobenius norm is *upper bounded* by the product of intermediate alphas, even ignoring further erasure. On all eligible positions in 48 real DEV 257-character passages, per-seed median `log10(product alpha)`:

| Seed | Lag64 | Lag192 |
|---|---:|---:|
|1701|-61.71|-185.83|
|1702|-56.99|-173.19|
|1703|-68.80|-209.58|
|1704|-47.30|-143.36|
|1705|-66.67|-201.68|

These are upper bounds on old-state influence, **not** retrieval probabilities or an end-to-end learned-accuracy proof. They are far below useful FP32 signal at 192 steps.

**Direct erasure on the 5 historical checkpoints:** preserve the exact same authentic prefix/suffix and absolute ordinals, but zero **only** the two aligned associative matrices at t=128, leaving native, basis, waves, and concept states untouched. All five seeds showed **exactly zero FP32 historical logit difference** and NLL difference for lag192 over 3,072 DEV source predictions/seed. This isolates aligned-memory nondependence rather than conflating all carried states.

**Historical loss gradient at the original source embedding output:** On one authentic DEV passage, each of five seeds, at both lag64 and lag192, source-t0 embedding-output gradient evaluated to zero FP32; recent t-1 gradient norms were nonzero (approximately 0.018–0.140). This is a 10-probe measurement, not a claim that all possible source-gradient values universally equal zero. In these trained examples there is no usable long-range learning signal.

## Why next-character NLL regressed

The ordinary source change versus original comprises weighted concept readout and coupled memory writes, not history-index correction. A **frozen checkpoint, zero-training** 2×2 numerical intervention independently changed readout (mean / weighted) and writes (independent / coupled):

| Pretrained source weights | Independent write + mean read | Coupled write + mean read | Coupled write + weighted read |
|---|---:|---:|---:|
| Original model weights | **2.53744** | **2.64935** | 2.64998 |
| v2 graph-on weights | 2.57059 | **2.56141** | 2.56277 |

Original weights suffer **+0.11192 DEV NLL** when coupled writing is switched on without retraining, versus a +0.00049 weighted-readout change under original writing. This identifies the much larger **frozen operator perturbation**, **not** a causal decomposition of the final retraining difference (+0.02534 NLL). After v2 training, weighted vs mean readout changes NLL by +0.00136; a separate same-weights graph-off lesion worsens NLL by +0.00033. Thus graph connectivity is real, but not sufficient for language superiority.

## Additional structural failure

`tac_wave_ced_500k_v2.py` forms the ordinary next-char aligned read with `ordinal_code(t-1)`. The historical lag64/192 read is passed only into `HistoricalWaveCED500K.history_head`; the retrieved historical head output is **not directly fused into the next-character decoder**. Any next-char improvement must come indirectly via shared weights. No source-owner ID exists in the carried state, so position legality is not source authorization.

History-trained five-seed DEV historical NLL:

| Lag | Correct source ordinal | Wrong shifted ordinal | Zero read | FIT unigram prior |
|---|---:|---:|---:|---:|
|64|4.39764|4.76775|4.77039|**3.53225**|
|192|**4.91637**|4.74488|4.77601|**3.63324**|

At lag192 the wrong key and zero read outperform correct-key recovery. **Correct chronological indexing is not correct-source retrieval.**

## Correct next steps, not retrospective rewrites

1. Preregister actual 2×2 **trained** real-text factorial (independent/coupled writes × mean/weighted readout) with five matched seeds, params/data/updates, full resource receipts and immutable negative prior; frozen interventions are not enough.
2. Restore a source-specific long-retention learning pathway: for 10% survival across 192 steps in a uniform scalar system, alpha must be at least ~0.988 (necessary, not sufficient in the presence of key-directed erasure); preserve a distinct rapid-update store. Avoid coupling all writes to globally shrinking old memory. Test bounded selective replacement/owner-indexed episodic storage, not uncontrolled growth.
3. Require a nonzero gradient from authentic source facts at >64/192/512 delays and direct, causal, source-authorized use of historical reads in the actual prediction head. Test old-source erasure, wrong-key, zero-read, wrong-owner and simple source-index retrieval.
4. Use independently held-out real source-ID data, fair RAG/SSM/GRU/Transformer/naive priors, cold repo-based original checkpoint replay and untouched TEST. Do not promote the model until actual retrieval and useful prediction exceed these controls.

**Evidence gap:** detailed per-seed diagnostic JSON and execution scripts are preserved in the separate conversation diagnostic archive, not yet checked into Git/CI; this record does not claim independent cold replay of the new diagnosis.
