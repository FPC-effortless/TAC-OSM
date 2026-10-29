# LANGPRETRAIN076 — two-free-T4 genuine 500K pretraining throughput measurement

**Status: BENCHMARK IMPLEMENTED, KAGGLE EXECUTION NOT SUBMITTED.** This draft is
stacked on LANGPRETRAIN075 PR #614 / LANGSCALE074 #613. No paid platform and
no independent generalization or model superiority claim are authorized.

## Question and exact study

Can the unchanged original LANGSCALE074 graph-free upper48 native
fast/slow/8×8/aligned/wave/concept language TAC, instantiated with 499623
actually trainable parameters, perform authenticated next-byte AdamW updates
on two physically observed Tesla T4s? What is the per-worker measured
**raw UTF-8 byte-token target throughput** (not BPE or word tokens)?

Freeze original public publisher PSF CPython 3.13.0 at commit
`60403a5409ff2c3f3b07dd2ca91a7a3e096839c7`,
original SOURCE023 receipt, 13 FIT sources + registered UTF8 FIT supplement.
EXPOSED DEV is never read by the worker. TEST never read.
Check the exact pinned `CONTEXT026` source SHA256 and `upper_reallocation.py`
Git blob. Preserve original full h48 499623p count and model geometry.
This is an **internal source-authenticated GPU efficiency pilot** only.
No capability, accuracy, memory scaling or CPU/GPU speedup comparison can be
inferred from a single run.

## Frozen measurement contract

- Seeds `1701` and `1702`; Kaggle accelerator exactly `NvidiaTeslaT4`
  **2 physical 16GB T4s**. One isolated process per T4, its own frozen seed.
- Per seed: 8 warmup AdamW updates + 32 measured AdamW updates, batch 8,
  128 original byte next-target positions per example = **40,960 total**
  genuine FIT target exposures and **32,768 timed** target exposures.
  These are not 40,960 unique source bytes.
- Fixed source-owned `scripts/langlayer035_sampling.py` deterministic
  owner/offset sampler, no cross-document example packing or state carry.
  Frozen optimizer AdamW lr .001 FP32, gradient norm clipping 1.
- End-to-end *timed* span includes source-window selection, Python tensor
  materialization, CPU-to-CUDA transfer, original full forward loss, backward,
  clip and optimizer step. Synchronize before and after the measured region,
  disable TF32. Report measured **byte targets/s** per worker and peak
  allocated CUDA tensor bytes. This is NOT pure model FLOP/s.
- Exact original source/sample SHA; model and optimizer/RNG checkpoint SHA;
  cold-reloaded fresh model logits on FIT input checked within fixed 1e-6
  absolute/relative error. Do not claim bitwise GPU equality.
- Hard per-worker bounded elapsed 600s and abort if GPU is not T4, missing
  seed→physical slot mapping, missing original source, changed model hash,
  unexpected parameter count, nonfinite losses or failed replay.
- Report wall and actual peak memory. Neither notebook creation nor a
  self-reported CUDA name independently establishes T4 hardware authenticity.
  Preserve incomplete/error receipts. No TEST, no hyperparameter tuning,
  no claims from synthetic tests.

### Result fields

`measured_byte_tokens_per_second = 32768 / measured_synced_wall_seconds`
for each seed worker separately. **Do not** add two rates unless both GPU
processes are confirmed overlapping; the aggregate rate is sum of both
completed measured targets divided by the verified paired effective wall
interval. Report genuine `GPU_T4_PAIR` allocation status and elapsed
wall, including failed/partial runs. Avoid presenting bytes/s as
tokenizer-specific language tokens/s for larger-vocabulary tokenizers.

## Kaggle packaging blocker (important)

The GitHub connector can write draft source files and read Actions evidence,
but does NOT currently provide workflow-dispatch or Kaggle submission.
The existing `scripts/stage_kaggle_research.py` packages a
`code_file=entrypoint.py` Kaggle kernel. Kaggle's documented `kernels push`
is not proof that arbitrary extra files in that directory, including
`workload.py` and the original source bundle, become available inside the
private runtime. The past no-fit smoke lacked verified output receipts.
**DO NOT submit this worker yet.** First inspect and test a reviewed
*single-file embedded, checksum-verified source bundle* (or a separately
authorized private Kaggle dataset attachment) under offline execution.
No internet access in the Kaggle notebook. Do not remove the original
no-fit or free-quota authorization safeguards for this convenience.

A completed source and runner packaging change must include a deliberate
`research/kaggle_workloads/` adapter, complete reviewed
`research/mathematical-engineering/langpretrain076/FAIR_COMPARISON.json`,
`research/phone_queue/langpretrain076.json` source and active
Kaggle dataset references, and actual per-file/checkpoint hashes.
The repository's protected `kaggle-research` environment must approve
any quota-consuming submission; the current dispatch workflow must be
reconciled with the **internal_research** vs independent-qualification
amendment rather than bypassed by an unverified flag. Verify remaining
free Kaggle minutes at submission, explicitly refuse pay-as-you-go and use
no alternate GPU. If free minutes unavailable, leave BLOCKED.

The operator explicitly requested this **$0** T4 microbenchmark, not
paid compute or full 10M/50M/100M pretraining. Avoid repeated smoke runs.

## Verification

The included unit tests check arithmetic, fixed seeds, source digest geometry
and fail-closed inputs on deliberately constructed no-fit fixtures. CI must
not run `run()`, access Kaggle credentials or create a real-trained
checkpoint. These tests are **not** GPU metrics. The actual Kaggle test remains
pending a fully reviewed source-bundling and quota-safe execution path.
