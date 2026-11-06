# CSL-125 R4 targeted memory precision — preregistration

2026-11-06. Parent PR718 commit eb296bfe19b36042d850379b393926f9d8eb8e49. Same R4 weights and exact-real recurrence, not an R5 architecture. No original source, failed receipt, fitted model or protected dataset is overwritten.

## Question and intervention
Can the inherited normalized-memory numerical failures be cleared by using FP64 only for the small numerator/mass scan, carried numerator/mass state and normalized read, while embeddings, convolutions, feed-forward/attention projections and learned weights stay FP32 (or their existing autocast execution precision)? No learned parameters are added. Do not assume this is efficient on GPUs or phones.

The proposed opt-in precision adapter replaces only R2 role-block implementations with source-equivalent blocks holding the SAME Parameter objects and state_dict names. It must reject old FP32 carried state; reset/replay is mandatory. New carried memory has an explicit type and stays FP64 through detach/continuation. The original R4 and all earlier source modules remain byte-identical.

## Frozen numerical comparisons
Reuse scripts/audit_csl_scan_cancellation.py's seven original fixtures, four chunks and three objectives: 84 rows. Inputs are original FP32 values, gradients are compared with the existing differentiable serial FP64 reference, gradient atol=5e-6 / rtol=5e-5, forward atol=1e-5 / rtol=1e-5. No dropped cases or changed tolerances. Original FP32 control is expected to retain 11/84 failing rows. The proposed FP64-memory operator naturally evaluates this diagnostic normalization/reduction in FP64; do not misrepresent that as clearing the FP32-objective arm or fixing the original source. Save all four argument gradients, forward differences and per-case decisions.

## Full-model checks
1. Exact learned parameter count, parameter-object identity, state_dict key/value equality at installation; full-size meta count 123873263 <125000000. No extra encoder, learned gate or tokenizer.
2. Independently written serial FP64-memory reference vs associative implementation on an actual small R4: compare outputs and every participating parameter gradient at atol5e-6/rtol5e-5. Unused auxiliary heads are reported, not treated as failures of tool SFT.
3. Full vs streamed chunks and future-target poison; unchanged stream ownership; reject old precision state; retain FP64 in detached carry. Frozen full-model streaming tolerance atol3e-4/rtol3e-5; report numerical values, never claim bitwise equivalence across chunk geometry.
4. Actual full 123873263-parameter FP32 model B1/T8, all exits and all head losses, forward/backward without optimizer. Record finite/nonzero tensor counts. Separately CPU BF16 autocast on reduced R4, finite gradients and precision boundary. CUDA AMP is NOT RUN unless physically available and independently approved.
5. Same-weight CPU one-thread B1/T64 restricted next-byte prefill: warmups2/repetitions7, alternating original/precision order at independently constructed 1M/10M/125M configurations. Report raw samples and medians, not a speed claim. Include carried-memory byte change. Full tool-generation/grammar and GPU/mobile energy remain NOT RUN.

## Scope and decisions
Engineering tests and random-weight diagnostics only. Model weights may be allocated for actual gradient tests but optimizer steps=0; GPU minutes=0; no corpus or TEST access, tokenizer fit, calibrated confidence, learned tool accuracy, runtime quantization, external effects, merge or deployment. Finite gradients are not evidence of learned competence. Record any failure and preserve it. Parent11/84 remains historical failure even if the opt-in path passes. A successful local/CI numerical repair does not substitute for source-qualified training/splits, signed independent evaluator/ACL evidence, budget approval or matched controls.

References: PyTorch numerical-accuracy and AMP documentation; established precision-island engineering, not novel math. Existing PNDS exact-real recurrence scope is unchanged; IEEE/full-source proof and independent qualification stay OPEN. User's125M cap and merged declared-target-v2 supersede stale100K-only ancestor prose. Next trained study must use the current policy and real source evidence, not inherited no-fit registrations.
