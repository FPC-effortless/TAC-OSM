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

## Kaggle deployment and the $0 operator gate

**Offline source packaging is implemented and tested (NOT executed on Kaggle).**
`scripts/langpretrain076_fit_only.py` verifies all authentic CPython publisher
FIT bytes and never loads exposed DEV/TEST raw documents. The deterministic
`scripts/langpretrain076_kaggle_bundle.py` builds a single code_file
`entrypoint.py` embedding 14 checked original FIT documents, model Python code,
and source verifier, with complete archive and member SHA256 checking before the
Kaggle notebook runs. No Kaggle datasets, internet, hidden developer credentials,
training, GPU session or external files are fetched during bundle creation.
The manifest `research/mathematical-engineering/langpretrain076/FAIR_COMPARISON.json`
and proposal `research/phone_queue/langpretrain076.json` pin the original
499623-weight h48 study, original publisher cohort/split, and 15-minute
bounded requested Kaggle allocation. The `source_bundle` exception is limited
to this exact study/workload; other studies still need their source references.

**Manual-only protected execution** is defined in
`.github/workflows/langpretrain076-free-t4.yml`. It requires the literal
`APPROVE_T4_X2_FREE_ONLY`, owner confirmation of remaining FREE Kaggle T4
quota, the existing protected `kaggle-research` environment, the enabled-GPU
secret and previously configured Kaggle credentials. It stages and verifies
original sources, then requests exactly `NvidiaTeslaT4` with a 900-second
remote execution cap. No paid hardware fallback, no PR-triggered automatic
GPU run, and no unauthorized TEST access.

**Actual external operational blockers remain:** this novel model research
PR is still draft/unmerged, the trusted manually dispatched workflow must
first be available on the default branch, the operator must approve the
protected free-quota execution, and Kaggle must accept and finish the notebook.
Connected GitHub actions used for this drafting do NOT expose workflow_dispatch
or a direct Kaggle execution capability. We cannot bypass those controls and
may not claim a run from staging alone. The original Kaggle smoke output was
incomplete; that does not invalidate the genuine source-bundling CI.

**Read-only evidence audit** is in
`.github/workflows/langpretrain076-read-only-audit.yml`. It retrieves only
the existing saved output for this exact private Kaggle kernel. The script
`scripts/langpretrain076_verify_t4.py` rejects absent/failed remote execution,
missing two workers, missing model checkpoints, altered SHAs, wrong original
worker code/source/parameter counts, incorrect actual T4 device slots and
invalid byte/s measurements. A passing digest is source-file integrity, not
independent physical hardware attestation or model capability validation.
Original trained checkpoint bytes remain private rather than being uploaded
into public GitHub CI artifacts. A separate long-lived original checkpoint
archive policy remains to be established before training beyond this pilot.

The operator authorized only a **$0** bounded T4 throughput microbenchmark,
not paid GPU or the full 10M/50M/100M campaign. If free quota is unavailable,
the run must not start.

## Verification

The included unit tests check arithmetic, fixed seeds, source digest geometry
and fail-closed inputs on deliberately constructed no-fit fixtures. CI must
not run `run()`, access Kaggle credentials or create a real-trained
checkpoint. These tests are **not** GPU metrics. The actual Kaggle GPU training test remains **NOT EXECUTED**. Its reviewed
source-bundling and protected workflow are implemented, but deployment approval,
trustworthy saved-run outputs and live free quota have not been observed here.
