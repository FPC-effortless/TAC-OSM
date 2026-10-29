# LANGSCALE075 — native and aligned associative address geometry (8→12→16): mixed/near-null result

**Read-only empiricism, authentic source and persistent evidence:** [six original CPU model fits](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40881468228) (all six trained, checked checkpoint SHA/optimizer/RNG/source order and cold exposed DEV replay); [original TAC study draft #615](https://github.com/FPC-effortless/TAC-OSM/pull/615). The original aggregate's *presentation printing* failed on stale field `upper_hidden_width`, **after writing the complete result JSON**; the first overall original workflow run is therefore **FAILED**, not retroactively rewritten. Separate [read-only recovery PR #616](https://github.com/FPC-effortless/TAC-OSM/pull/616) [Actions #40881916911](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40881916911) [verified full result artifact #12899899056](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40881916911/artifacts/12899899056) passed. **ZERO retraining** in the recovery; downloaded all six original 075 trained checkpoint and receipt artifacts, validated per-file SHA256, source owner/source order, optimizer+RNG receipt and original [070 SHA-verified baseline #40871491842](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40871491842). [PNDS native math companion draft #73](https://github.com/FPC-effortless/pnds-lean/pull/73) [Lean CI #40881418157](https://github.com/FPC-effortless/pnds-lean/actions/runs/40881418157) **PASS** for limited exact geometry and parameter counts only.

## Data, model and controls

- Source-authentic publisher-licensed PSF CPython 3.13.0 bytes, 14 FIT owners (13 original+Unicode supplement), frozen 4 already EXPOSED DEV owners. AdamW lr0.001, 160 real original fit steps ×8 owner-separated 129 byte windows with 128 next-byte targets each = **163840 genuine publisher source outcomes per new fit**, seeds1701/1702/1703; actual native m12/m16 checkpoint weights and optimizer/RNG cold replay. **Original m8 baseline is *not* retrained**, reused original LANGSCALE070 p500k trained seed checkpoints and source/sample-order SHA.
- Exact original `CONTEXT026` k8 mapped source, original h48 upper recurrence, FFN455, first-stage fast/slow output dimensions88/88, wave8×2 and concepts6×16; all nofit original k8 source parameter/forward/trace/gradient parity constructed tests passed.
- New matrix k12 and k16 active native k×k and 2 aligned tables 16×k. New native key/value and learned k→8 native readout; **more than one component changes**. Active useful learned model weights 499623,500175,500663 (<0.21% gap), no dummy weights or size padding. Training FLOPs are NOT matched or even measured across k; CPU process cost reports efficiency only. No k>8 incremental serialization/carry is qualified by this study.

## Three seed mean EXPOSED DEV next-byte metrics

| Native matrix | Active parameters | Mean NLL ↓ | Top1 ↑ | CPU process fit seconds |
|---|---:|---:|---:|---:|
| **m8 (original 070)** | 499623 | 2.31701444496 | **38.1348%** | **21.5024** |
| m12 | 500175 | 2.32549394084 | 38.0290% | 25.1092 |
| m16 | 500663 | **2.31655337226** | 38.0615% | 28.3081 |

Paired seed NLL, **new native dimension minus original m8**; negative = improvement:
- m12: seed1701 **+0.01044112**, 1702 **+0.01449522**, 1703 **+0.00050215**; all **3/3 worse**. Mean +0.00847950 nats/byte.
- m16: seed1701 **+0.00654081**, 1702 **−0.00215093**, 1703 **−0.00577310**; **2/3 improved**, mean **−0.00046107** nats/byte. This tiny 0.00046 mean effect is not statistically reliable or an architecture gain.
- m12 versus original m8 mean CPU seconds +16.8%, m16 +31.7%, with host timing confounds. Do not infer mobile speed or power.
- Mean top1 actually **decreases** slightly at both higher sizes despite m16's very small mean NLL decrease.

**Interpretation:** training the wider native+aligned associative geometry on the same short genuine text windows did **NOT** establish a reproducible 500K parameter memory-scaling win. m12 hurts all three seeds; m16 mixed/near null. There is no actual long-range information-retention proof here, as every training/evaluation source window resets. Do not promote the k16 candidate based on a favorable 0.00046 NLL mean. This is a different negative scientific result from LANGSCALE074's upper-recurrence failure: 075 targets the native matrix but also changes aligned addresses/readout. Further capacity inflation without long-history demand is unlikely to be justified by these data alone. An independent original long-streaming state test is the proper next decision gate.

**Scientific scope:** original genuine source but DEV previously exposed, 3 seeds only, training FLOPs unequal and ~500K trained parameter sizes within 0.21% parity, unchanged protected TEST; source-to-PNDS complete PyTorch IEEE/AD/serial-vs-parallel exactness OPEN. All PRs remain draft/unmerged. Zero new GPU/Kaggle quota, no checkpoint promotion or sealed TEST access.
