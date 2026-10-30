# LANG10M002 — authentic-corpus and matched 10M controls (NO FIT)

**Status:** prospective no-fit source and architecture preparation. This study
does not launch a Kaggle job, refit the tokenizer, use a model optimizer,
read protected TEST, or claim LFM/Mamba/Transformer competitive performance.

Sources inherited: completed CODEVALUE004 (PR #657) is immutable and shows
that broader data helped 500K TAC while all utility screens failed.
LANG10M001 (PR #658) retains its 4K tokenizer and dual-T4 capacity reports.
Its exposed no-update FP32 batch2×1024 profile measured 26.6K tokens/sec,
not optimizer-inclusive train throughput. Unscaled AMP zero-gradient tensors
mean the initial training reference must be FP32 unless a separate loss-scaled
full-gradient parity experiment clears it.

## Independent control definitions

| Arm | Original trainable=deployed parameters | Mechanism |
| --- | ---: | --- |
| TAC LANG10M001 | 10,001,188 | Original first-stage TAC memory, four causal upper blocks |
| GRU | 9,995,660 | Two PyTorch GRU layers, 806 hidden coordinates |
| Simple diagonal selective SSM | 10,002,288 | Four 256-wide diagonal, input-selective update + SwiGLU layers |
| Causal Transformer | 10,001,940 | Six 256-wide pre-LN attention blocks, four heads, FFN1966 |

All controls have a 4096-token vocabulary and return [B,T,4096].
The SSM control is a deliberately simple diagonal SSM, *not* Mamba or LFM3.
The Transformer uses a learned 1024-position embedding, not long-context
RoPE. The GRU/SSM have no explicit pos embedding. These choices must be
documented as functional, not merely parameter-count, differences.

All four architectures are untrained. Exact model counts must be tested
on instantiated modules; a paper count alone is insufficient. The native
simple SSM currently loops in Python across sequence positions, so speed
claims would be invalid without optimized reference kernels. The Transformer
has quadratic causal attention; no edge superiority is inferred.

## Corpus scale and composition (PROPOSAL ONLY)

- Pilot tokenizer source: 219 authentic code/docs FIT files, ~1.17M
  BPE tokens and 3.63MB text. Do not train a 10M foundation model
  repeatedly on this small amount of largely code-specific data.
- Phase A: seek at least 50M *token instances after file dedup* for an initial
  genuine-language scaling experiment; this threshold is a planning gate,
  NOT a sufficient-data scientific proof.
- Phase B: aspirational 200M training-token budget (~20 tokens/parameter
  heuristic, **not** a TAC scaling law); actual unique/text and epoch
  counts must be measured separately.
- Proposed mixture to register/freeze before sampling:
  approximately 60% educational/general-language, 25% permissively sourced
  code, and 15% high-quality technical/scientific explanations. The
  best mixture is unknown. Report unique document counts and token
  positions, and repetitions/epochs separately.
- Use **publisher-separated** FIT, SELECT_DEV, TRANSFER_DEV and a fully
  access-separated TEST cohort. Select checkpoints ONLY from SELECT_DEV,
  then freeze one checkpoint per architecture/seed before TRANSFER_DEV.
- Evaluate both code and natural-language publishers, and measure
  bits per original UTF-8 byte. A tokenizer trained on mostly Python
  may inadequately encode ordinary prose. Retokenize a new BPE4096 only on
  the larger mixed FIT, and freeze tokenizer SHA/normalizer/special IDs
  for every model before any comparison; preserve tokenizer V1 intact.
- Decontaminate benchmark prompts and near-duplicates, remove exact
  duplicates, detect sensitive data/PII, honor opt-outs, pin source
  snapshots, and archive original source/license receipts. Existing local
  admission gate proves hashes and disjoint publisher metadata only;
  it does NOT perform near-dedup/PII/licensing legal review.
- Avoid storing raw multi-gigabyte corpora/checkpoints in normal Git PRs.
  Use private, hash-addressed external packages and immutable manifests.

**Candidate authentic sources to audit individually** (NOT acquired or
approved by this PR):
1. FineWeb-Edu educational English web data (dataset card lists ODC-BY):
   https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
2. Dolma/Dolma 3 diverse general-language corpus (dataset-level ODC-BY
   does not erase original-source obligations):
   https://docs.allenai.org/training_data/dolma
3. The Stack v2 code dataset (per-repository source licenses and opt-outs;
   bulk access has additional Software Heritage/INRIA terms, so do not
   assume it is freely bulk-downloadable):
   https://huggingface.co/datasets/bigcode/the-stack-v2

**Important source format limitation:** initial corpus_gate.py requires a
40-hex Git source commit and original Git blob checksum. This is suitable
for independently pinned source repositories. A separate original-corpus
snapshot/record provenance adapter is needed before onboarding web-corpus
shards from FineWeb/Dolma. Do NOT fabricate Git metadata to make them fit.
Any external dataset must be locally materialized, individually traced to
a permitted source/license, and separately cleared.

## No-fit data admission

    python research/language-scaling/lang10m/corpus_gate.py \
      --manifest /authorized/local/originals.json \
      --tokenizer-model /authorized/local/lang10m-bpe4096.model \
      --output /fresh/output/corpus-admission.json

The manifest is locally authored and must have schema "original-source-v1",
study "LANG10M002" and a sources array. Each row requires:

    {
      "partition": "FIT",
      "kind": "code",
      "publisher_group": "independent-company-or-project",
      "repo": "owner/repo",
      "original_path": "src/module.py",
      "source_commit": "40-lowercase-hex",
      "git_blob_sha1": "40-lowercase-hex",
      "sha256": "64-lowercase-hex",
      "local_path": "materialized/module.py",
      "license_id": "SPDX-or-reviewed-identifier",
      "license_path": "licenses/license.txt",
      "license_sha256": "64-lowercase-hex"
    }

Source and notice paths must exist locally and match hashes. The validator
checks UTF-8, distinct files/blobs, publisher disjointness across the three
allowed partitions, source/notice bytes and vocabulary count if supplied.
It does not open protected TEST, fetch external content, fit a tokenizer or
train any model. Tests use made-up files as engineering fixtures only.

The admission ledger must stay OPEN for source rights, near duplicates,
entity contamination, PII and the complete protected TEST boundary. A
50M token-instance reading is not an automatic training authorization.

## Future matched training (NOT implemented / NOT authorized)

1. Freeze all original source/data, versioned tokenizer, document normalization,
   sample windows, context length, all 4 exact parameter counts and seeds.
2. Choose registered equal data/exposure, updates, optimizer schedule/tuning
   budget and DEV-only checkpoint selection. Measure FLOPs and CPU/GPU time
   separately; training FLOPs must be accounted before declaring superiority.
3. Validate full-graph per-module gradient viability (not only finite gradients),
   optimizer state/precision, causal prefix, original cold checkpoint replay,
   module state/owner resets, and source split immutability.
4. Run short pilot only with separately accepted private dual-T4 quota, physical
   UUID checks and no unsafe dtype fallbacks. Full checkpoints include optimizer,
   RNG, sampler and source hashes, never only weights.
5. Independently score publisher transfer after all checkpoints sealed;
   preserve failures and previous study evidence. No commercial claim from
   language-model loss alone, and no LFM3-scale claim at 10M.

Mathematical obligations: PNDS #94 covers only partial upper TAC recurrence.
The full TAC model and the three added controls have no new composed Lean +
Python/source correspondence. Mark OPEN pending separate per-variant record,
not autoqualified by unit/fixture tests.

Run no-fit checks:

    python -m pytest -q \
      research/language-scaling/lang10m/test_controls_and_corpus.py

No GPU allocated or used by these checks.
