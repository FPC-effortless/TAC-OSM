# LANGSPEED034 — trained model decoding-speed protocol

**Registered 2026-10-29, before first actual GitHub Actions latency comparison.**

## Primary question

How fast are the **actual trained, originally checkpointed, present-day** 500K byte language predictors on the **same GitHub Actions x86 Ubuntu CPU runner**, at inference time—not training time?

Three unchanged neural arms from the original LANGCOMPARE031 five-seed research (no Mamba-3): source-pinned TAC-CONTEXT026 (500122p), official Pathway BDH (501504p) and the causal Transformer (496960p). One original seed, 1701, is sufficient to compare operations/hardware under identical architecture dimensions; it is **not** a five-run CPU performance distribution. Include FIT-only P3 and pre-registered fixed .75 P3/TAC hybrid as reference predictors, not matched-param neural comparisons.

## Metric definition

Run on one physical GitHub Actions ubuntu-24.04 job, CPU-only PyTorch 2.10.0, NumPy 2.3.5, two intra-op CPU threads, one inter-op thread, eager float32 with eval()+inference_mode(), batch size 1. Warm up before timing. Never touch DEV/TEST or use GPU. Input prompts are real, original-source-owner FIT bytes. The same prompt length and generated token count apply to every arm. Pin, SHA256-verify and cold-reload all original trained seed-1701 model+optimizer/RNG checkpoint files from frozen run 40780215655; load the BDH original Git blob at pinned upstream commit, no substitutes.

Two **separately reported** measurements:

1. **Prefill full forward**: neural models only, context 16/64/128 bytes, each 1 untimed warmup call and 3 timed repeated full forwards, report median/min/max wall ms. This creates all positions' logits; it is **NOT** incremental decode speed. P3 has no comparable learned neural full-forward tensor, do not fabricate one.
2. **Actual greedy generation**: prompts with 32 or 128 original bytes, 4 new bytes, batch=1, greedy argmax per step, for every step recompute using **the last at most 128 bytes**, 3 timed generation repetitions with arm order rotated. Report median ms/byte, min/max ms/byte, corresponding approximate bytes/sec and raw individual per-step timing with output SHA. Initial prompt-prefill work is *excluded from the generation-only timer*, since the first output is generated from a full-prefix call in the first measured step. This is an execution-model measurement with repeated forward and bounded rolling context, not a measurement of an optimal streaming recurrent decoder.

None of the three current callable model APIs is proven to support production persistent incremental state or Transformer KV caching in this protocol. The CONTEXT026 forward explicitly rejects nonzero carried start; therefore performance here measures an **unoptimized full-prefix fallback**, not hypothetical O(1)-memory TAC decode or optimized Transformer cached decode. Future implementations should add and test identical-logit incremental interfaces against whole-prefix semantics before any optimized architecture speed comparison.

## Fairness and limits

- Same runner/hardware, exact trained original checkpoints, same input owner, same precision, batch and thread counts. Comparisons among neural models at ~500K learned parameters (within 1%).
- Generation may choose different next bytes per predictor; counts and context lengths are fixed. P3 is CPU Python probability lookup; this is the exact source implementation, not optimized production C++ counts.
- The hybrid mixes 25% TAC probability with 75% P3 probability and pays both inference costs, including 256-way softmax/mixture.
- Single benchmark runner may be noisy due to shared-host CPU frequency; retain raw timings and avoid statistical significance claims.
- Track speed separately from predictive quality. The prior DEV NLL 2.0997 TAC/2.5746 BDH/2.5600 Transformer (all five seeds), and hybrid 1.7779 (exposed DEV), come from **other studies**, not this speed experiment.
- Do not claim GPU, mobile, cloud deployment, CUDA kernels or optimal streaming speed; no hardware used except current shared x86 GitHub CPU host. No parameter training, TEST access or promotion.


## CPU runner safety amendment before successful measurement (2026-10-29)

The first two complete same-host jobs were forcibly stopped by GitHub-hosted runner shutdown, leaving **no valid latency receipt**. In the revised prospective no-fit execution, **only the three neural models TAC-CONTEXT026, official BDH and causal Transformer are timed**. Building the full-FIT P3 n-gram table is removed because it is a separate CPU-intensive preparation step, unrelated to neural decoding and suspected of consuming the hosted-runner execution envelope. The original P3/hybrid baseline and all accuracy results remain unchanged in their own prior studies. Their decode speed is **NOT RUN**, not zero. The retained protocol measures batch-1 neural prefill at 16/64/128 bytes (3 timed calls each) and actual 4-byte greedy autoregressive decoding for 32/128 byte prompts (3 repetitions). Original SHA-verification, checkpoint cold replay, source, runner, threads, no new training and no cache/streaming limitation remain unchanged. Future P3/hybrid CPU latency requires an isolated separately timed source-count preparation and inference-only stage. All aborted attempts are preserved in Actions history.


## Short-form measurement amendment (before successful latency receipts)

Three GitHub-hosted runner jobs were forcibly shutdown during the larger neural speed workload, without valid metrics or Python exceptions (original attempts retained in Actions run history). To avoid opaque retries, the immediate **short-form pilot** is now exactly: two neural prefill context lengths (32 and 128 bytes), one timed forward after one warmup per arm/length; two autoregressive greedy bytes each at prompt 32 and 128 (so full prefix computation for each byte), **two complete generation repetitions** per arm, all three original trained models executed sequentially on the **same physical GitHub runner**. This gives coarse implementation-level latency, not robust throughput quantiles. The original larger protocol remains recorded above as a future benchmark when CI runner runtime is stable; no historical result erased. Even if successful, no optimized recurrent or KV-cache speed claim is allowed.
