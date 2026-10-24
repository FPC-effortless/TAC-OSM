# ESWAM011 — complete-model mathematical engineering results

**Status:** Software equivalence and formal structural safety **PASS**. Predictive intelligence/commercial superiority **NOT ESTABLISHED**; no new training or real prediction gate run. No checkpoint is promoted. Candidate remains draft and unmerged.

## Actual engineering implementation
- `src/tac_osm/variants/vectorized_exchange_world.py` replaces the ORIGINAL `ExchangeMemory` per-batch Python read/write loops with fixed-slot batched masked similarities, normalized softmax, deterministic exact source-ID admission, protected-role writes and learned symmetric/antisymmetric merge gating.
- `VectorizedExchangeWorldActionModel` retains the original complete encoder, innovation correction, ordered fast→slow recurrence, readout and physical/action heads with identical parameter naming/count.
- `VectorizedSelectiveExchangeWorldModel` retains **all** ESWAM003 causal observed-velocity, movement probability, adaptive shrinkage and logged command-head math. It changes only the memory operations. Original 101226 trainable active parameters remain 101226; candidate model weight loading uses strict original `state_dict` compatibility.
- A conditional batch-level merge check preserves the original autograd semantic difference between absent gradient and explicit zero gradient when no merge is authorized. This matters for optimizer weight decay and momentum. There are **no per-sample Python loops** when batch >1.
- This is parallelism over batch and fixed memory slots, NOT parallelization of the genuinely sequential nonlinear fast→slow or cross-time recurrence. No new simultaneous-traffic-agent representation was invented.

## Verified Python gates
- Original and vectorized all evidence-role reads and writes match for batches 1, 3, 16, repeated IDs, no-hit, all-disabled, overflow refusal, matching-slot reuse, all three roles.
- Full original world-model multistep state, mean, scale, advisory action, flags and backward parameter gradients match at batch 1, 4, 12.
- Full preferred ESWAM003 real-dimension forward dynamics, movement estimates, physical outputs and backward parameter gradients match at batch 1 and 4.
- **All five** original Git-owned **authentically trained** ESWAM003 seed-76001..76005 checkpoint weights strictly load into both models and pass no-fit multi-step forward state/prediction parity tests. These probes are diagnostics; they do not establish improved authentic real-world forecasting.
- Repository full **1160 tests PASS** and documentation-integrity test count up to date. Original workflow CI: https://github.com/FPC-effortless/TAC-OSM/actions/runs/40079811237 .

## Verified Lean kernel
- Pinned Lean v4.25.0 CI: https://github.com/FPC-effortless/TAC-OSM/actions/runs/40079811220 **PASS**.
- `WholeModelStep.lean` encodes source causality/dependency ordering of **all** sections: input+mask+logged action+time; encoder; source-key prediction retrieval; verification; fast then slow recurrence; factual observed memory write; query/read; decoder; mean and scale; advisory action; unverified prediction write; and returned persistent state.
- Lean proves separate observed/predicted/verified role store invariants **and whole_step_cannot_rewrite_verified** across the full typed transition.
- Lean's operator maps are abstract **typed functions**, not a formal proof of the concrete nonlinear PyTorch linear algebra, floating-point arithmetic, autograd, numerical stability, exchange-gate semantics, data preprocessing, movement head or trained performance. These proof gaps are not silently closed.

## Whole forward CPU timing
Workflow https://github.com/FPC-effortless/TAC-OSM/actions/runs/40079533383 **PASS**. Same CPU runner, PyTorch 2.14.1+cu130, one thread, same model weights, two recurrent steps per timing sample, 20 raw repetitions per architecture and batch. Original and vectorized model maximum output difference 2.98e-8 across measured batches.

| Batch | Original median | Vectorized median | Original / vectorized |
|---:|---:|---:|---:|
| 1 | 1.18 ms | 1.56 ms | 0.76× (vectorized slower) |
| 8 | 3.21 ms | 1.77 ms | 1.81× |
| 32 | 9.84 ms | 2.15 ms | 4.57× |
| 128 | 35.23 ms | 3.32 ms | **10.61×** |

Unrounded exact data and **every raw repetition** are preserved in `CPU_BENCHMARK_RESULTS.json`; throughput is a diagnostic, not a claim of lower training FLOPs, objective architectural superiority or higher predictive intelligence.

## Scientific gates still failed/unexecuted
This code is *mathematically equivalent*, so any authentic predictive errors of the original ESWAM003 remain; equivalence cannot improve intelligence. ESWAM004's stationary/calibration failures also remain factual. ESWAM007 required real multi-agent 256-frame future prediction and matched parameter/data/training-budget GRU and causal-attention controls; those experiments have **not** been implemented/trained by this PR. The original ESWAM006 instantaneous-velocity physical baseline remains stronger than long indiscriminate history prior on the one-site original traffic source. An additional source location, actual multi-actor overlap eligibility, trained entity-specific memory and paired actor confidence intervals remain compulsory before a claim of real superiority or commercial value.
