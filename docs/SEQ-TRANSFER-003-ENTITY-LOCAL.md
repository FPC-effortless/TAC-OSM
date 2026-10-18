# SEQ-TRANSFER-003 — Entity-local TAC-OSM sequence modelling

**Date:** 2026-10-18. **Status:** Completed local frozen-weight test; out-of-domain supervised readout transfer; no promotion, no trained checkpoint changes.

## Scientific conclusion

DEV-061.5 `split_robust` trained entity-local content + processed entity features is decodable for short-lag memory better than its single random-weight control. **Mean across 14 synthetic task–length cells: 87.48% trained vs 81.91% random initialization, vs 82.93% independently trained no-CASM.** DEV-062-R1 TAC **81.62%** scarcely differs from its scratch random frozen-weight control **81.45%**, no-CASM **81.32%** or GRU **81.55%**. A simple task-trained **5-gram averages 93.0%**; there is no state-of-the-art predictive model claim.

The 96/128/256-bit memory probes indicate strong *recent-bit decodability* for DEV061.5, **not storage or recall of the entire past**: the exported memory is only the most recent 64 accepted records, and first-bit retrieval is at chance by length 96. The native models cannot directly output these predicted bits; a new independently trained linear head makes these predictions.

## Full matched frozen-probe benchmark

All arms used exact same DEV058 predecessor synthetic `create()` tasks and generator seeds. Each task and length has 600 train / 400 held-out examples, binary labels paired with complements. A new StandardScaler + L2 logistic regression (`C=0.1`, liblinear) is fit only on that task's train examples. There are 7 tasks × 2 lengths × 8 model arms. The pretrained parameters remain frozen. Only the task-specific linear head is fitted; random arms are frozen random weights with the same memory capacity and *forced admission*.

| Model | 32 events | 96 events | Mean 14 cells | 32→96 mean |
|---|---|---|---|---|
| 0615_entity | 87.5% | 87.4% | 87.5% | 86.2% |
| 0615_noCASM | 83.4% | 82.4% | 82.9% | 81.7% |
| 0615_random | 82.9% | 80.9% | 81.9% | 82.1% |
| 062_TAC | 80.7% | 80.7% | 80.7% | 73.6% |
| 062_R1_TAC | 81.5% | 81.7% | 81.6% | 61.1% |
| 062_R1_noCASM | 80.8% | 81.9% | 81.3% | 65.5% |
| 062_R1_GRU | 81.8% | 81.3% | 81.6% | 78.5% |
| 062_R1_random | 82.1% | 80.8% | 81.4% | 50.0% |
| 5-gram | — | — | 93.0% | — |


| Task at 96 events | DEV061.5 | DEV061.5 random | DEV062 | DEV062-R1 | 5-gram |
|---|---:|---:|---:|---:|---:|
| constant | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| alternating | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| period3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| period5 | 88.2% | 78.5% | 73.5% | 73.2% | 100.0% |
| delay3 | 98.5% | 71.0% | 71.0% | 77.8% | 100.0% |
| delay5 | 75.5% | 67.8% | 67.8% | 66.2% | 100.0% |
| iid | 49.8% | 49.0% | 52.5% | 54.8% | 51.2% |


DEV061.5 uses 128 learned `node+content` features for the first 2 entities, R1 uses a 32-dimensional global learned latent; the head feature counts, training targets and pretrained task distributions differ. The comparison is NOT parameter- or feature-matched across architectures, though each model's own random / no-CASM controls offer more informative context. Older DEV058 study probes 1568 raw-memory values and therefore is not a direct architectural ranking.

## Entity-local interleaving test

Random event stream alternates subject A and B (48 events per entity for L96, truncated to 64 total events by the native runtime). New supervised linear heads predict each entity's latest, 3-back and 5-back bits, and XOR of the two latest bits. Trained on 600, tested on 400 fresh independent sequences. The input supplies the two entity IDs via the native public key manifest, so entity discovery is **not** learned.

| Held-out task, 96 events | DEV061.5 entity-local | DEV061.5 random | DEV062-R1 TAC |
|---|---:|---:|---:|
| A_last | 100.0% | 100.0% | 96.5% |
| B_last | 100.0% | 100.0% | 99.8% |
| A_lag3 | 92.5% | 78.8% | 67.2% |
| B_lag3 | 91.0% | 69.2% | 62.5% |
| A_lag5 | 73.8% | 72.2% | 65.2% |
| B_lag5 | 74.8% | 69.2% | 68.0% |
| cross_xor | 100.0% | 82.2% | 83.2% |


The trained DEV061.5 node representation permits **100% XOR readout** in this synthetic test, but the separately trained no-CASM checkpoint ALSO gets 100% with combined node and local content. This does **not** isolate CASM as the cause of the computation. Repeating data seeds 5 times yielded DEV061.5 average 100% XOR and 94.3% A-lag3 on L96; content-only XOR remained 48.2%, while node-only XOR was 100%. This distinction indicates features beyond raw local content carry nonlinear interactions.

## Retention and sequence length

Both exported model families retain at most 64 **raw public event records** before running recurrent/latent processing. 96+, 128 and 256 event streams all discard the earlier records. On fixed L256 tests (600 train, 400 held out), the trained readout gives:

| Retrieval at L=256 | DEV061.5 trained | DEV061.5 random | DEV062-R1 TAC |
|---|---:|---:|---:|
| first | 51.2% | 50.5% | 50.2% |
| last | 100.0% | 100.0% | 100.0% |
| lag3 | 96.5% | 71.5% | 81.5% |
| lag5 | 79.0% | 74.2% | 71.2% |
| lag16 | 55.0% | 57.5% | 59.0% |
| lag32 | 50.0% | 49.2% | 49.0% |


These are independently trained probes at each length, not trained on 32 and evaluated at 256. The earlier 32→96 evaluation **without refitting** averaged 86.25% DEV061.5, 61.07% DEV062-R1 and 50.0% random R1 (macro across the 7 tasks). Full cell scores in `results.json`. Longer input length does not mean the model consumed all historical events.

## Robustness, statistical limitations and remaining control

Independent IID next-bit labels have no learnable pattern. A larger new 6,000-example check gives 50.65% DEV061.5, 49.70% DEV062-R1 (both approximately chance). Single 400-example IID scores, up to 55%+, are NOT evidence of predicting random bits.

The input uses a repeated synthetic public status record: fixed event type index 0 with a positive trained admission gate; each bit is a public state value under a fixed entity identity, and events carry a public clock. This is out-of-distribution for the maintenance-trained weights. Both initial and alternative variants were checked against original public JSONL demo choices and original SHA256. A direct vectorized read of the last 64 records was verified bit-for-bit equivalent to replaying 96 events via the actual `observe()` method for all trained selected arms; random arms force admission to remove untrained gate collapse.

The studies use first preregistered export for each model/arm, not the full 20-seed research cohorts. Data-seed repeats only measure test-set variation, not training-seed variability. No original DEV-061 (before DEV-061.5) binary weights were available from the captured CI artifact; therefore this study directly executes the accessible **DEV061.5**, **DEV062**, **DEV062-R1** exported entity-aware descendants, and does not claim to have run DEV061 original.

A separately attempted 30-epoch scratch supervised GRU comparison **timed out before completing the protocol**, so its partial output is excluded from model rankings. An end-to-end, compute-matched newly trained TAC vs GRU/SSM/Transformer comparison on length extrapolation and symbolic/counterfactual sequence tasks remains unexecuted.

## Provenance

Original learned portable exports were recovered from existing successful GitHub Actions artifact archives of TAC-OSM draft PRs #220 and #221. All exact weights and inference model source included in the standalone reproducibility archive.

- **0615_entity**: SHA-256 `7acc30f5e492b0be599a17ae770a10fd8e04108e179d4f781b7d4b2c636f88e7`; replay [4, 4]
- **0615_noCASM**: SHA-256 `60298d4379e3d749e8ed7cd98170ac615724b7b048a0d77bf1fabf98d3dd5726`; replay [4, 4]
- **062_TAC**: SHA-256 `abc16d85af2c1d289fd7eff61e2883e26427c4f983e89a56f9320dc507015019`; replay [4, 4]
- **062_R1_TAC**: SHA-256 `06d6a747c4402d68c89cb59fdfced60178c594d9763fcffa0725d4875881e57f`; replay [5, 5]
- **062_R1_noCASM**: SHA-256 `9b29964dadeeb04e9f7ff2397498ba77a1de468b7a22aaf38edcd6ea13a7318e`; replay [5, 5]
- **062_R1_GRU**: SHA-256 `d6a0c2467b9f805c3ee005a15c54d69233bd7d7128ae231664db18086fe220a8`; replay [5, 5]

Executed with Python `3.13.5`, PyTorch `2.10.0+cpu`, scikit-learn `1.8.0` and NumPy `2.3.5` (CPU; one thread).

## Run

```bash
pip install torch numpy scikit-learn
python probe.py                       # original exported demonstration replay
python experiment.py                  # full 7-task x 2-length matched probe
python extra_tests.py                 # memory retention, 2-entity, IID robustness
python ablation.py                    # 5 data-seed repeats, node/content
```

The excluded, unfinished supervised GRU attempt is in `scratch_gru.py` and `scratch_gru.log` for transparency; do not treat it as a completed benchmark. No teacher, task-specific hand-coded solver or future-label feature is provided to the frozen TAC model.

**Research-only.** These results do not certify autonomous next-token generation, original CDL symbolic memory, C5 scaling, commercially qualified foresight or learned entity discovery.