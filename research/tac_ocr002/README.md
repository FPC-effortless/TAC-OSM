# TAC-OCR-002 — actual OCR data acquisition

This supplies real image/transcription pairs for the **first recognition pilot**
of TAC-OCR-001. It does not fit a model or establish recognition accuracy.
The neural architecture and original trained research are unchanged.

## Source and scope

Publisher: [CORD, Park et al.](https://github.com/clovaai/cord), original
[CORD v2 dataset](https://huggingface.co/datasets/naver-clova-ix/cord-v2),
immutable revision `7f0115a4b758a71d6473b8d085751692da2fef98`, CC-BY-4.0.
Original split assignments are 800 training receipts and 100 validation
receipts. This workflow never downloads the 100-receipt TEST shard.

CORD contains Indonesian receipts. It supplies selected annotated fields,
word quadrilaterals and text, not a complete transcription of all visible
page content. Recognition using publisher-provided crops is **conditional
on annotated boxes**; it cannot establish end-to-end page OCR, region
detection, reading order, long-document capability, or Nigerian Pidgin quality.
Annotation category/group labels are not predictor inputs.

## Acquire and prepare

Install Pillow, PyArrow and huggingface-hub. For the optional PyTorch batch
loader/tests, use the same PyTorch runtime as TAC-OCR-001.

```bash
python scripts/phone_lab.py plan research/phone_queue/tac-ocr002-source-audit.json
python -m research.tac_ocr002.prepare --download --raw /path/cord-original --output /path/cord-recognition
python -m research.tac_ocr002.audit /path/cord-recognition
python -m research.tac_ocr002.curate --raw /path/cord-original --prepared /path/cord-recognition
PYTHONPATH=src:. python -m pytest -q research/tac_ocr002
```

The original five train/validation Parquet shards are SHA256/size-verified
against pinned publisher objects. Combined source download is approximately
2.07 GB. Originals, crop images, labels and feature caches belong outside
Git. The only source files committed here are code, provenance metadata,
protocol and compact actual audit receipts.

Preparation reads one receipt record at a time and writes crop PNGs and
JSONL records incrementally. Content-addressed filenames reduce duplicate
storage and carry no annotation text. Crop pixels are RGB rectangles around
the original quadrilaterals, padded two pixels, clipped to the source image.
Within a publisher field, words are grouped by publisher `row_id` and joined
in their original order with one space. Case, Unicode, punctuation and
spelling are preserved. This grouping handles fields that span several rows.
Missing row IDs, clamped geometry, nearly blank crops and exclusions are
recorded, never silently declared correct.

The initial source audit stopped at the validation boundary because the
publisher uses `meta.split="valid"`, whereas its documentation says `dev`
and the Hugging Face shard says `validation`. All 100 validation metadata
records were inspected: this is a vocabulary alias, not a resplit. The
original protocol and failure are preserved. The revised parser accepts
those three validation aliases and still rejects a training/validation
contradiction. No model outcome was used to change the cohort.

## Corpus and loader

The prepared package includes `records.jsonl`, RGB `crops/`, original
`annotations/`, document provenance, `audit.sqlite`, duplicate findings,
character inventories, publisher source/license and preparation receipts.
`audit.py` independently verifies all prepared text against original words,
all crop hashes/dimensions/modes and all document/split parent links. A
12-region contact sheet supports sampled visual inspection; this does not
establish the correctness of every human annotation.

```python
from research.tac_ocr002.dataset import RegionDataset, collate_regions
data = RegionDataset('/path/cord-recognition', split='train')
batch = collate_regions([data[0], data[1]])
# Only batch['images'] goes to the frozen vision encoder.
# Decoder inputs: batch['input_ids']; supervised targets: batch['labels'].
# IDs and flags in batch['metadata'] stay outside the neural predictor.
```

The correct teacher-forcing shift is unambiguous:

`ids = [BOS] + UTF8(text) + [EOS]`

`inputs = ids[:-1]`; `labels = ids[1:]`.

This preserves the final printed character and the EOS prediction. Padding
targets use `-100`, excluded from cross-entropy. The encoder sees image pixels,
never the target string. Reset state between unrelated receipts; any
same-document carried-state training needs its own declared sequence policy.

## Quality and qualification

Inspect the actual compact receipts beside this README and full `AUDIT.json`
in the prepared package. Source-image overlap, near-duplicate candidates,
exact crop overlaps and conflicting transcriptions require explicit review.
Repeated short labels such as TOTAL are not automatically evidence of a
repeated receipt. Original rows and splits are retained for inspection.

The actual source contains **11,307 training regions** and **1,244 validation
regions**, derived from 800/100 receipts. All five original file hashes and
every source image/annotation pair were verified. Prepared-data checks cover
all crop hashes, shapes, RGB modes, parent links and source text order.
Twelve sampled crops were visually inspected; this is not a full human audit.

The audit found 77 exact crop patterns shared across splits, 10 crop patterns
with conflicting text, 15 clipped source boxes and 12 perceptual near-image
candidate pairs. Whole-source duplicates are zero for both encoded bytes
and decoded RGB pixels. Shared short crops do not necessarily mean that
entire receipts were duplicated.

A declared train-only filter removes 108 regions affected by overlaps,
conflicts or flags. `train_filtered.jsonl` contains **11,199 regions**;
`validation_original.jsonl` preserves all **1,244 validation rows byte-for-byte**.
The loader prefers those manifests when present. Original rows and all
excluded training IDs/reasons remain available. Near-image candidates and
conflicting validation labels still require adjudication. This filtered view
does not qualify a scientific training preflight.

The transcriptions contain **zero naira characters**; Nigerian Pidgin coverage
is unestablished. Neither is implied by a UTF-8 output vocabulary.

No TEST payload was accessed by this job. This is not a claim that no public
web example or pretrained model has ever exposed benchmark content; that
contamination question remains unaudited. A fresh independent heldout cohort
is needed for strong final quality claims.

This package does not qualify the signed training preflight. The actual
source chain, split policy, data/tuning/compute parity, protected evaluator
and independent source/outcome/reward/ACL evidence remain separate gates.

Next data phases: genuine Nigerian printed/Pidgin image-text pairs; layout
and reading-order data such as DocLayNet; table/formula data as separately
licensed tasks; authentic long documents. Evaluation benchmarks such as
OmniDocBench should stay outside the training mixture.
