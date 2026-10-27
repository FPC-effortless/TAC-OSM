# TAC-LANG-MLAG-001: genuine-source paired 64/96 retrieval training

**Status: exploratory; independent TEST still CLOSED.** Executed locally against the original authenticated Vim 9.1 user manual FIT/DEV partition and all ten original TAC-LANG-GATE-003 checkpoints. This is a within-architecture training-objective experiment, **NOT** a head-to-head against Transformer, Mamba, GRU, RWKV, or RAG. Results do not demonstrate general language competence, multi-session recall, or reliable inference beyond the 128-character development context.

## Change, and strict parity

Original architecture is unchanged: 100,680 active trained parameters, native 8×8 correction state, two 16×8 source-ordinal associative banks, two-headed source routing, trained `multiscale` positional mode, and the same SiLU-vs-raw-linear (archive label `tanh`) gate variants. No new learned memory or information sidecar.

Original objective: `L_char + 0.30 L_history(64)`.

New objective: **`L_char + 0.15 L_history(64) + 0.15 L_history(96)`**.

Both run 120 AdamW updates, batch=8, character context=128, LR=.002, weight decay=.01, clip norm=1, same five paired seeds 1701–1705, same 30 FIT and six DEV whole-document splits, same byte-to-ASCII mapping, and *bit-identical per-seed input sampling SHA256* against archived originals. Across arms, source-model Python SHA256 is `835212d74cc0d2d4d22ff7e75beb12771586cce58f2e63943690e5d859417457`; frozen original protocol SHA256 is `181936efb2bd7513597180455079590289ccbf9150381bea5625f489faecb2f1`. All 36 source-document hashes and ten archived checkpoint hashes were enforced before fitting.

**Training compute is NOT equal:** the new arm adds a second historical read and CE term; parameter counts, training sample order, optimizer-update budget and source/eval splits are matched. Record actual time and computational work; do not call this an equal-FLOP comparison.

## Primary results, authentic document-disjoint but previously seen DEV

Mean across 10 checkpoints (two gate modes and five paired seeds); NLL in nats/character, lower is better:

| Metric | Original lag64-only training | Paired 64+96 training | Change |
|---|---:|---:|---:|
| Next-character NLL | 2.824216 | 2.827443 | +0.003226 (worse) |
| Lag64 historical NLL | 3.460926 | 3.378885 | -0.082041 (better) |
| Lag96 historical NLL | **5.219261** | **3.127606** | **-2.091655 (40.1% reduction)** |
| Lag64 historical exact-character accuracy | 15.729% | 19.222% | +3.493 percentage points |
| Lag96 historical exact-character accuracy | 5.514% | 24.740% | +19.225 percentage points |

All **10 of 10** paired checkpoint comparisons improved lag96 NLL. Across the five independent *seed blocks* (paired gate modes on the same seed), original-minus-new lag96 NLL benefits ranged 1.6309 to 2.4167 nats. This is meaningful but does not create a protected test result from a reused DEV split.

Detailed historical-loss per checkpoint (nats):

| Seed | Gate | Original lag96 | New lag96 |
|---:|---|---:|---:|
| 1701 | raw linear | 5.310060 | 3.310641 |
| 1701 | SiLU | 5.296279 | 3.261412 |
| 1702 | raw linear | 4.411229 | 2.759712 |
| 1702 | SiLU | 4.344900 | 2.734567 |
| 1703 | raw linear | 5.823529 | 3.593396 |
| 1703 | SiLU | 5.832794 | 3.598019 |
| 1704 | raw linear | 5.395383 | 2.946246 |
| 1704 | SiLU | 5.295303 | 2.910959 |
| 1705 | raw linear | 5.276548 | 3.105580 |
| 1705 | SiLU | 5.206586 | 3.055527 |

### Memory-use controls; no retraining, same saved weights

| Historical lag | Source original correct key | New correct key | New FIT-only unigram | New zero read | New wrong key +1 | New wrong key +8 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 3.460926 | 3.378885 | 3.531907 | 4.852031 | 3.489985 | 3.766060 |
| 96 | 5.219261 | **3.127606** | 3.542343 | 4.852031 | 3.222506 | 7.629065 |

The +1/+8 wrong-address figures use the **same targets** only at positions where both query delays are valid; therefore their sample count is smaller than the correct-key NLL. The comparison is a **diagnostic** of query specificity, not an equal-sample direct aggregate. The +1 penalty at lag96 is modest (~0.095 nats); the +8 penalty is larger. Better short-history source addressing should not be mistaken for general relational or semantic memory. The zero read follows the existing history head on zeros and is not a separately retrained no-memory baseline.

The new lag96 correct-key result also beats the previously useful **FIT unigram 3.5423426** on these DEV windows; hence purely replacing TAC with a corpus-level prior does not explain its improvement. Actual long-range usefulness on unseen material remains to be proven.

### Operational evidence and files

The **10 new binary checkpoint files** were saved with learned weights, AdamW state, Torch/NumPy/Python RNG states, source/dataset protocol and exact per-seed sampler hash. Every checkpoint was immediately reloaded into a fresh model and had its full measured DEV metrics reproduced within 1e-6. The local evidence archive contains checkpoint SHA256 manifest, `RESULTS.json`, `MEMORY_CONTROLS.json`, the original source-model file, and the exact training/replay scripts. Bundle size ~11.46 MB, SHA256 **`b19e026e49c06093850fafa6ca63bffd84f1153aa947f8cae67938d66a4c5bfd`**. The downloadable bundle is supplied in the ChatGPT research conversation.

**Important repository gap:** binary checkpoints have not yet been pushed into GitHub, and the matching source manual is not a public pinned GitHub CI fixture. Thus the user's strict **repo-hosted checkpoint/cold CI replay promotion gate remains OPEN** despite local no-fit replay. Do not merge or promote this arm as qualified until a lasting binary repository storage mechanism and independent cold CI replay exist.

Run once the original archive and byte-identical Vim9.1 manual are legitimately provided:
```bash
python train_multilag_real_fit.py --archive /path/to/original_archive \
  --source /path/to/vim91/doc --out /path/to/mlag001_output

python audit_multilag_memory_controls.py --archive /path/to/original_archive \
  --source /path/to/vim91/doc --out /path/to/mlag001_output
```

The repo's normal CI runs **constructed unit diagnostics only** and MUST NOT claim real data replay without the original source and checkpoints.

## Meaning of this result

The earlier lag96 failure was not evidence that the architecture fundamentally cannot preserve those past events. Retargeting the same capacity and update budget at both 64 and 96 produced a large DEV improvement. It does **not** remove the deeper capacity, aliasing and interference limitations; nor does it show correctness beyond 128 characters or a throughput/cost win. It is an objective-learning result rather than a new model-family invention.

The native assistant still needs better next-character fluency; the slight NLL regression is a warning against optimizing retrieval alone.

## Next qualifying experiment

1. Freeze this two-horizon objective and its metrics, then obtain **new real-text whole-document sources** with distinct independent TEST, lawful use, stable byte hashes, no contamination and source identity. Keep any fresh TEST sealed until all candidates are frozen.
2. Train at the same ~100k learned parameter budget and identical updates/splits against a matched GRU, attention, Mamba/RWKV-like recurrent baseline when actually supported, plus strong unigram/bigram and retrieval sidecar baselines. Explicitly document any impossible matched FLOP/hardware comparisons.
3. Register longer 256/512/2048 event memory queries, alias/stale fact updates, wrong-key and no-memory controls, per-entity chronology, abstention and calibration; do not simulate these tasks as primary evidence.
4. Demonstrate actual online streaming state parity, training throughput, per-token CPU/GPU inference cost, energy and external memory I/O. Distinguish the 408-float neural state from external source records.
5. Archive every fitted checkpoint, code, optimizer, RNG, data authorization, sample hash and CI replay before considering superiority or commercial claims.
