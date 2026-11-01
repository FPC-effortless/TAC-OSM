# TAC-OCR-003 — private Kaggle grounding pilot

This is a **single-architecture, two-seed training registration**, not a
TAC/Transformer/GRU comparison. The objective is the first real transcription
fit of the unchanged TAC-OCR-001 TAC decoder. No model has been fitted by
source preparation or software tests. Training remains blocked until the
repository's independently signed prefit gate passes.

The operator requested Kaggle training and supplied credentials on 2026-11-01.
The authenticated Kaggle read succeeded. Credentials are used only by the
host SDK, outside source bundles, datasets, notebook code and logs. This
authorization does not create independent scientific source/access evidence.

## Registered workload

- Source: TAC-OCR-002's exact 216 MB CORD region archive; 11,199 retained
  training regions, all 1,244 original validation regions; no TEST payload.
- Model: frozen 21,812,352-parameter TrOCR-small **vision encoder** and
  1,309,376-parameter TAC-OCR-001 decoder; 23,121,728 deployed parameters.
  Its graph is unchanged from source `d2700f9b3864dab0d71a59a96643bbb72635f144`.
  This does not reuse the pretrained TrOCR text decoder as the TAC decoder.
- Efficiency: encode each unique crop once across two GPU shards; store
  pooled 128x384 FP32 image-only features in disk-backed arrays; both seeds
  reuse that cache. Length buckets reduce padding. No encoder gradients,
  AMP, quantization, compilation, remote downloads or changing architecture.
- Exactly two physical T4s, seed 1101 on GPU 0 and seed 1102 on GPU 1.
  Fifteen complete passes, batch 64, AdamW, fixed final-epoch selection.
- Evaluation starts from BOS and image features. The predictor accepts no
  gold text or target lengths. Report original case-sensitive Unicode CER,
  whitespace WER, exact regions, missing EOS, time and CUDA peak allocation.
- Retain every epoch's model, optimizer, RNG and schedule cursor; immutable
  source/data/protocol identities enable replay. Preserve logs and failures.
- Session timeout: 7,200 seconds, at most 240 allocated GPU-minutes for
  two T4s. Internal workload deadline: 6,300 seconds. No hardware fallback
  or unrequested retry. Actual FLOPs remain unmeasured; no equal-compute
  comparative claim is permitted.

## Mandatory external prerequisite

[AGENTS.md](../../AGENTS.md) and the
[independent runner gate](../REWARD-INDEPENDENT-RUNNER-GATE-2026-10-26.md)
require an operator-pinned public verification key and an access-separated
assessor's signed source/goal/scorer/actual ACL findings for BOTH variant-seed
rows **before any optimizer operation**. Neither a Kaggle API key nor an
agent-written JSON supplies this evidence. Never generate a signing key in
this workflow. Missing or invalid evidence blocks before CUDA work.

On Kaggle the notebook reads the public key and signed prefit receipt from
operator-controlled Secrets named `TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX`
and `TAC_OSM_PREFIT_ATTESTATION_JSON`. Independent postfit evidence must bind
actual checkpoint hashes through `TAC_OSM_POSTFIT_ATTESTATION_JSON`; run
`python -m research.tac_ocr003.assess --output OUTPUT` after the assessor has
issued it. Fitting and independently qualifying a result are separate stages.

Source qualification is also OPEN: 12 perceptual near-image candidate pairs
still require adjudication, and conflicting validation annotations remain
unaltered. No preflight attestation should ignore these unresolved findings.
The CORD corpus has no demonstrated Pidgin or naira coverage and does not
establish complete-page detection or long-document OCR.

## Offline staging and replay

`stage.py` packages only explicit Git-tracked source from an immutable commit,
the hash-pinned data ZIP, exact publisher checkpoint and dependency wheels.
The initial notebook locates an exact-hash asset manifest, verifies every
archive, rejects unsafe ZIP paths, installs wheels without networking, and
invokes the signed gate before extracting image payloads or using CUDA.
Use the official Kaggle SDK's `kernels_push(..., no_run=True)` to save it
without execution. An accepted save is not a fit or a quota allocation.

Training entry point, only after genuine signed preflight is provided:

```sh
python -m research.tac_ocr003.train --data DATA --output OUTPUT
```

For interrupted fits, keep original outputs; mount the old output read-only
and supply `--resume PREVIOUS_OUTPUT` to a newly authorized allocation. Each
seed restores `last.pt` (the last complete epoch); no extra retry is automatic.
The full final checkpoint needs the separately pinned encoder distribution
to provide image-to-text inference. The trained model is not a detector.

## Verification scope

`test_training.py` exercises signed-gate denial before optimizer/CUDA work,
byte/EOS alignment, deterministic all-row batching, source mutation rejection,
checkpoint identity/replay, hardware rejection, Unicode metrics and actual
graph inference invariance under owner renaming. Tests make no optimizer
updates. Fixtures are software evidence, not OCR skill or source attestation.
The PNDS record is the unchanged TAC-OCR-001 record in
[PR105](https://github.com/FPC-effortless/pnds-lean/pull/105); full-model
forward/backward and finite-precision proof obligations remain OPEN.

The training objective is mean negative log probability of the next original
UTF-8 byte or EOS over nonpadding targets. Inputs are BOS followed by original
bytes through the penultimate prediction position. The original final printed
byte is retained. Padded targets have index -100 and no loss contribution;
causal masking prevents later padding from changing earlier predictions.
Each region's recurrent state is freshly initialized, so this pilot cannot
establish cross-page memory or long-document retention.

Kaggle automatically expanded the first uploaded `.zip` inputs. Provider
read-back detected this before any execution. The corrected package stores
unchanged ZIP bytes under `.bin` names so input archives retain their exact
hashes. The initial private dataset/notebook versions are preserved; neither
executed. This is a packaging correction, not an architecture or data change.
