# LANGPRETRAIN075 — byte-language pretraining qualification campaign

**Status: SOURCE/COST DESIGN ONLY — NO NEW MODEL FITS.**
Parent: [LANGSCALE074](https://github.com/FPC-effortless/TAC-OSM/pull/613),
original head SHA `e9b68535cb900d84e74dbbe012583e710dac9d10`.
This draft must remain unmerged until reviewed. No real publisher corpus has
been downloaded/qualified in this work. No Kaggle minutes, protected TEST,
paid cloud costs, model checkpoints, or source credentials were touched.

## Falsifiable research objective

Can the exact source-pinned graph-free layered native TAC byte model continue
to improve held-out authentic next-byte NLL as its **per-seed** byte exposure
increases from 10M to 50M to 100M, and is its realized training plus inference
quality/cost frontier preferable to an equally trained non-TAC baseline?

The original LANGCOMPARE031 CPU benchmark measured 500122 trainable weights
and mean process time 44.17 s for 327680 byte targets per model/seed, 320
updates. These are genuine original publisher FIT bytes, but they are **not**
a long-run or GPU throughput estimate. LANGSCALE074 uses an actual original
499623-parameter h48 model at the 500K owner-exception budget; first-stage
native fast/slow/8x8 and other original components must stay exact.
Select/freeze one model source and checkpoint identity **before** a fit;
the latest draft is not automatically approved for pretraining.

## Corpus and licensing

Targets (planning proportions, **not** evidence that any dataset is eligible):
web 50%, code 25%, technical prose 15%, dialogue 10%.
Potential inputs: selectively reviewed FineWeb-Edu, permissively licensed
publisher source repositories, openly licensed scientific manuals and
documented conversation corpora. Never silently assume a Hugging Face, Stack
or web dataset is licensed for every purpose; retain per-document legal review
and publisher release, timestamps, license terms, source URL, exact raw SHA256.
No bulk download or provider login is part of this PR.

Stage local immutable raw UTF-8 documents with a JSON manifest:
```json
{
  "schema": "tac-osm.byte-pretrain-corpus.v1",
  "documents": [
    {
      "id": "publisher-doc-001",
      "split": "FIT",
      "category": "technical",
      "path": "technical/doc-001.txt",
      "sha256": "<independently measured 64-character lowercase SHA256>",
      "license": "<reviewed source license>",
      "publisher": "<original publisher>",
      "source_url": "https://<actual publisher document>"
    }
  ]
}
```
Also provide separate DEV documents. The example deliberately contains
placeholders, not invented real publisher identities, licenses, or hashes.
The audit requires `FIT` and `DEV` source documents and rejects `TEST`.
Do not put originals, source manifests with sensitive locations or credentials
in PR artifacts. Approve only publicly redistributable dataset material.

```sh
python scripts/langpretrain075_audit.py audit --manifest /private/corpus/manifest.json --corpus-root /private/corpus
python scripts/langpretrain075_audit.py cost --exposure-bytes 10000000
# ONLY AFTER a measured end-to-end two-T4 aggregate FIT throughput exists:
python scripts/langpretrain075_audit.py cost --exposure-bytes 10000000 --measured-aggregate-bytes-per-second 20000 --usd-per-gpu-hour 0
```

Audit covers strict UTF-8, exact whole-document dedup, per-document windows,
files/path containment, byte SHA256 and split exclusion. It does **not** establish
near-duplicate removal, true publisher authenticity, full licensing compliance,
independent training authorization, learned model quality, or external evaluation.
No cross-document training window and no unintended persistent state carrying
from one independent publisher document to another: explicit boundary reset
versus separately registered learned boundary is a prospective controlled
condition, never quietly combined.

## Frozen candidate stages (conditional, not scheduled)

| Stage | Byte targets **per seed** | Objective / advancing gate |
|---|---:|---|
| 0 | 0 | Authenticate original sources, fix models, train/DEV separation, seed schedule and paired baseline |
| 1 | 10,000,000 | Measure genuine end-to-end T4 training bytes/sec, model/optimizer/RNG/source hashes and strict cold replay |
| 2 | 50,000,000 | Continue only with independently reviewed positive learning trend and no leakage/regression |
| 3 | 100,000,000 | Continue only with qualified earlier checkpoints, fixed evaluation and acceptable costs |

Two distinct frozen seeds at minimum, with one physical T4 assigned to each,
**never** claim 32GB combined memory from two independent 16GB devices.
Every model/seed gets the declared full exposure; with two seeds the total
processed byte-target exposures are twice the per-seed row.
Use exact source-owned tokenization (raw UTF-8 bytes, 256 logits) with source
owner boundaries and original byte-relative prediction time.
Require pinned training sampler, tokenizer, optimizer, LR, batch size, schedule,
seeds, source partition hashes, no future labels in inputs, identical exposure,
comparable trainable/deployed parameter count <=1% mismatch, training FLOPs
matched or explicitly exploratory, test sealed, and same source-prefix DEV
conditions. Compare n-gram, selective SSM, GRU and Transformer as available,
not just TAC ablations. Re-measure actual parameter count at prefit/postfit;
retain the owner-authorized 500K exception as scoped to language only.

**Primary metrics:** held-out DEV byte NLL/BPB, top-1, and independent long
context retrieval control at fixed source histories. **Cost:** source bytes
and exact training target exposure, process/GPU time (with CUDA synchronize),
actual peak memory, throughput including setup and checkpoint overhead,
physical accelerator count, per-seed evaluation time, incremental decode
latency and state footprint. **Integrity:** SHA256 of original source, order
and model/optimizer/RNG checkpoint, fresh-object replay, anti-cheat logs and
preserved failures. Historical DEV exposure is not fresh held-out TEST.

Fail stage promotion if missing original-source license/authorization,
publisher/reviewer receipts, source leakage, a failed replay, unexpected
architecture shift, unmatched comparators, an unknown throughput field,
or any signed evidence the selected execution lane requires.
No adaptive changes using protected TEST.

## Compute and financial gates

Use GitHub Actions **only** for brief no-fit diagnostics and approved bounded
CPU tasks. Kaggle dual physically observed NvidiaTeslaT4 may be used for a
justified CUDA task only after the repo's protected approval mechanism and
independent prefit checks. The queue proposal is intentionally **BLOCKED**
because it has neither a qualified source manifest nor a reviewed workload.
It DOES NOT authorize GPU runs or constitute a registration passing parity.

Maximum Kaggle minutes **per separately approved submission**: 120 requested
here, below the repo's 240-minute ceiling. This is a cap, not an estimate of
total training time, and does not authorize the entire 100M campaign.
**Paid-cloud authorization = $0.** Any future illustrative $25 contingency
requires a separate explicit permission, never implicit in this proposal.

The cost tool requires **measured aggregate end-to-end throughput across both
concurrent seed shards**; otherwise GPU cost/duration intentionally remains
null. Its CPU extrapolation based on short historic process time is explicitly
nonpredictive.

## Prerequisite follow-on changes, not attempted in this PR

1. Preserve publisher source bytes, independent source/license attestation,
   row/document-level deterministic FIT/DEV split and near-duplicate audit.
2. Register complete `FAIR_COMPARISON.json` under
   `research/mathematical-engineering/langpretrain075/` with true, pinned
   source digests, actual parameter counts and exact training/FLOP budgets.
3. Register complete dual-T4 Kaggle training workload with physically verified
   two-device seed-sharding and checkpoint replay, measured first-stage rate.
4. Run `scripts/phone_lab.py plan` and scientific/anti-cheat/preflight
   checks **before** any optimizer step; preserve the original evaluator and
   separate protected TEST.
5. Submit evidence and actual cost receipts to Issue #472. Do not merge this
   research PR automatically or call a source-only audit a model gain.
