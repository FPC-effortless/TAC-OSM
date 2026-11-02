# TAC-OCR-001 — efficient streaming OCR prototype

**Status: working pretrained OCR pipeline + untrained custom neural candidates.**
No custom OCR model has been fitted. This prototype does not establish
Unlimited OCR 2 accuracy or efficiency. The pipeline's usable recognizer is
Tesseract; the TAC/Transformer/GRU graphs are separately testable candidates.
Historical TAC results and source were not changed.

## Run now

Python 3.11+, Pillow, and an installed Tesseract 5 with the requested languages
are sufficient for images. PDF support additionally needs PyMuPDF. PyMuPDF
is AGPL/commercial licensed; review the applicable distribution license
before integrating it into a proprietary product. The code has no remote
OCR API and does not upload input documents.

```bash
python -m pip install pillow pymupdf
# Debian/Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-eng

python -m research.tac_ocr001 transcribe input.pdf --output outputs/book --workers 2
python -m research.tac_ocr001 transcribe input.pdf --output outputs/book --workers 2 --resume
python -m research.tac_ocr001 search outputs/book "invoice total"
```

Run from the repository root. The CLI supports PDF, common image formats,
and multipage TIFF. Outputs are `pages.jsonl` (text, per-word coordinates,
engine confidence and timing), `document.md`, an on-disk full-text index,
and `run.json`. Page numbering and the append journal support crash recovery;
resume refuses changed input, configuration, engine version or language weights.
OCR confidence is explicitly uncalibrated. Low-confidence/empty pages are
flagged for review. Failed OCR jobs fail the run rather than disappearing.

Default `--mode ocr` recognizes image pixels. Explicit `--mode auto` can use
a PDF text layer on pages with extractable text and no images; the output
labels that source. Native text is not counted as OCR model recognition.
Complex reading order, tables, formulas and handwriting remain unqualified.

## Where efficiency comes from

- Pages are rendered lazily. At most `workers` rendered/submitted page jobs
  are in flight; pixel limits cap the size of each render (PDF edge rounding
  can add a small number of pixels). Output is streamed to disk.
- Each OCR worker uses one OpenMP thread; outer page concurrency is bounded
  to avoid nested thread oversubscription. PGM stdin avoids intermediate
  PNG encoding and file round trips. Tesseract is still launched once per
  page; startup cost is included in end-to-end measurements.
- Text and search history are stored on disk. Search returns at most 16
  snippets and a requested UTF-8 byte budget. This search API is not passed
  into custom neural logits and is not described as learned visual routing.
- The candidate encoder runs once per line/region. Visual features are
  pooled to at most 128 references; shared visual K/V projections are
  cached once and reused across layers/tokens.
- TAC has a 128-token local window and fixed fast/slow associative state.
  The initial candidate disables the separate rotary memory branch and
  uses rank-one writes. Local RoPE remains present. It reuses actual
  LANGCACHE104 `Block104` and its chunk-scan/recurrent kernels.
- The optional frozen steady-state CPU graph removes Python traversal on
  filled-cache, one-token inference. It preserves source weights and clock
  as a tensor input. It requires fixed state/reference geometry and is
  rejected after parameter versions change. Compilation cost and trace
  warnings are reported. TorchScript is deprecated by the installed PyTorch;
  this optional path remains an experimental optimization, not a required
  deployment dependency. No CUDA/AMP/INT8 performance is claimed.

Bounded **working tensors** do not mean constant storage for an entire
document. The original document, output, feature/index files and PDF reader
metadata grow with page count. During training, autograd retains its needed
graph; inference cache bounds are not full-training-memory guarantees.

## Custom grounded decoder

`GroundedOCRDecoder` accepts previous UTF-8 bytes and image features, not
gold text embedded into vision features. Every layer applies cached visual
cross-attention, its recurrent/local core, and an active feed-forward path.
Stream ownership prevents state/reference reuse between documents. Owner
and page-ID strings are metadata only and never enter neural embeddings.

| Candidate | Layers / width | Active decoder parameters | FFN width |
| --- | --- | ---: | ---: |
| TAC / LANGCACHE104 blocks | 4 / 128 | 1,309,376 | 384 |
| Cached RoPE Transformer | 4 / 128 | 1,309,312 | 520 |
| Residual stacked GRU | 4 / 128 | 1,309,824 | 433 |

These are **decoder counts**, not total vision-language model counts. The
actual frozen encoder must be included in deployed counts. No dummy padding
matches counts; the feed-forward parameters are on the prediction path.
Parameter matching alone is not a scientifically qualified trained comparison.

`FrozenTrOCREncoder` optionally loads only the encoder retained from an
immutable `microsoft/trocr-small-printed` publisher checkpoint. It handles
DeiT's CLS and distillation tokens before assigning patch coordinates. The
publisher model is designed for single text lines: resizing a dense whole
page to its input resolution would discard small text. Region detection,
proper line crops and their accuracy must be evaluated before using this
candidate for full-page recognition. Initial raster pooling is a label-free
compression baseline; it does not guarantee retention of every character.

```python
import torch
from research.tac_ocr001.model import GroundedOCRDecoder
from research.tac_ocr001.presets import matched_presets

model = GroundedOCRDecoder(matched_presets()["tac"])
# encoder_features[B,N,384] and boxes[B,N,4] come from image pixels only.
refs = model.prepare_visual(encoder_features, boxes,
                            owners=("document-1",), page_ids=("page-1-region-1",))
logits, state = model(previous_byte_ids, refs, owners=("document-1",))
```

The forward graph is trainable. Random-weight logits are not OCR outputs.
The operational CLI therefore exposes only the pretrained practical backend,
not an untrained neural transcription option.

## Verification and profiling

```bash
PYTHONPATH=src:. python -m pytest -q research/tac_ocr001 tests/test_research_probe_agnostic.py
python -m research.tac_ocr001 profile --output results/CPU_PROFILE.json --steps 64 --threads 1
python -m research.tac_ocr001.smoke --output results/repeated-page-smoke
```

The profile uses random weights and constructed feature/token fixtures;
timings exclude the vision encoder. The repeated-page smoke uses one
attributed upstream image repeated across 1/10/50/100 PDF pages. Neither
is a real OCR quality benchmark. See `../tac_ocr001_results/` for original
measurements, failed memory-observation history and explicit unknowns.

Focused checks cover UTF-8 fidelity, image-sensitive graph connections,
future-token poison, every-parameter gradient connectivity, FP64 chunk/step
logit and named-state equivalence, batch/document isolation, compact physical
cache allocations, a scalar cross-attention reference, a sampled derivative,
frozen-step correspondence, true image recognition, TIFF/PDF reading, bounded
backpressure, search limits, worker errors and crash/resume behavior.

## Research boundary and next required evidence

`PROTOCOL.json` preregisters engineering scope and the proposed genuine-data
study. It is deliberately not a valid `FAIR_COMPARISON.json`, and does not
authorize fitting. CORD v2 is the proposed initial real receipt grounding
source, with publisher receipt-level splits preserved. A separate authentic
long-document study must test reading order, missing/repeated text, tables,
formulas and cross-page cases. Do not use repeated images as that evidence.

Before any optimizer step: authenticate original image/label data and splits,
measure the actual retained encoder and decoder, freeze a new scale-v2 parity
registration and measured training/tuning budget, obtain independent signed
source/outcome/reward/ACL evidence, pass the repository preflight, and prepare
the approved private offline dual-T4 workload. This branch does not launch
training or consume Kaggle quota. A new full training runner and data staging
still require that source-qualified registration; they are not supplied as
an unchecked fit command.

The reciprocal PNDS mathematical record is partial: new visual-attention
equations and source tests do not prove the full encoder/core graph,
floating-point stability or preservation of arbitrary document information.
Compilation and whole-model proof obligations retain their actual status.

Prior art: [TrOCR](https://arxiv.org/abs/2109.10282),
[Unlimited OCR Works 2](https://arxiv.org/abs/2610.39308),
[LANGCACHE104](https://github.com/FPC-effortless/TAC-OSM/pull/684),
[CORD publisher](https://github.com/clovaai/cord),
and [Tesseract performance guidance](https://tesseract-ocr.github.io/tessdoc/FAQ).

Implementation review: [TAC-OSM #690](https://github.com/FPC-effortless/TAC-OSM/pull/690). Mathematical companion: [PNDS #105](https://github.com/FPC-effortless/pnds-lean/pull/105).

## Draft state/cache lifecycle repair (TAC-OCR-004)

Decoder states and visual caches bind to one local model/load generation.
Successful, failed and parent-module `load_state_dict` calls invalidate earlier
state and references before loading tensors. Detached states retain ownership;
legacy untagged states must be rebuilt. The optional steady session preserves
the tag and is rebuilt after loading, including `assign=True`. The public
configuration signature and neural tensor graph remain unchanged. Tags are
trusted local metadata, not cryptographic checkpoint identity. Arbitrary direct
in-place parameter mutation is outside the eager generation guarantee.

See `LIFECYCLE_REPAIR.json` for no-fit receipts. The original reciprocal PNDS
source maps are immutable and intentionally reject the repaired source bytes.
A prospective reciprocal PNDS source/math binding and separate exact-head
review remain required before integration; these checks do not authorize fit.
