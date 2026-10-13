# DEV-011 — Frozen TAC-OSM persistent-state decoder probes

**Classification: DEVELOPMENT-ONLY; not preregistered, not a state-of-the-art world model, and not a success claim.** Previous DEV-010 full results were ~50% decision accuracy despite the memory state responding to historical changes. The causal question is whether the state contains recoverable target information that the readout fails to extract. A failed probe **does not prove** information is absent.

## Immutable parent source and checkpoint provenance

Pin exact [DEV-010 Actions run 38362662234](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38362662234), full artifact ID 11795975699 (`DEV-010-RAW-NOT-CONFIRMATORY`), SHA256 `a15685af62122a876c9d990fd877c060c22e41f2d81d80bd47f2426ab558fc56`. Use archived `native-seed{43,59,71}.pt` and `ced-seed{43,59,71}.pt`, verify each against the recorded checkpoint and parameter hashes in parent `report.json`. **Do not fine-tune, backpropagate through or update any parent writer weights**. The reference untrained writer must reproduce each native model's *initial* DEV-010 hash, not a guessed random initialization. This is a development re-analysis of a previously inspected experimental family.

## Question and contrasted controls

Can a fresh supervised probe recover a past payload (XOR current bit) from the fixed 4×8×24 = 768-scalar learned memory state, under noisy addressing and intervening competing writes?

For each writer seed, test:
- **native_trained:** trained DEV-010 TAC-OSM persistent DeltaMemory.
- **ced_trained:** writer component from trained DEV-010 model with CED reader; only the recurrent writer generates state.
- **native_untrained:** the original DEV-010 *randomly initialized* writer with an otherwise identical state capacity.

For each **frozen writer**, fit on paired publicly visible histories:
- **keyed_linear:** query-conditioned expert delta read (`state @ keys(query)`) + current public inputs → 2-way linear classifier.
- **keyed_mlp:** identical keyed features → 64-unit MLP → 2-way decision.
- **state_mlp:** full 768 recurrent-state scalars + query/current → 64-unit MLP. This is a deliberately higher-capacity *information-extractability* upper diagnostic, not a capacity-matched architecture baseline.

Also report:
- **Zero-memory reset** using the *same probe weights* with all memory-derived feature dimensions set to 0; in balanced opposite-past pairs this must score exactly 50% if the data interface is leak-free.
- **Swap-state control** swaps each matched pair's memory features while preserving their identical public present; genuine history-dependent decisions should change. An observed state L2 difference alone is not evidence of useful target decoding.
- **Full-history Euclidean nearest-address control** for context; it reads every stored example at query time and its O(H+D) addressing cost must be reported.
- **Retained vs acquired past** and near-key competing writes; histories of H=32/128 plus H=512 stress and D=4/16/32.

Inputs to the **probe** are exclusively recurrent memory, public noisy query vector and public current bit. No oracle target slot, evaluator role, target, gold address or full history enters probe inference. Offline training does use evaluator target classification labels, which is explicitly a different supervision regime from DEV-010.

## Conditions and measurement rules

- Three seeds 43/59/71; each writer seeded from an audited DEV-010 checkpoint. Train on 128 synthetic opposite-past pairs per seed/writer with matched settings H=32/128, D=4/16, near noise and retention/acquisition. Split by deterministic fully visible fingerprints and guarantee zero train/eval overlap.
- 240 AdamW probe steps at 0.002, **all writer parameters frozen**. The three probe kinds for an origin see exactly the same training episodes and minibatch schedule.
- Test on two deterministic held-out replicates of the 40 registered DEV-010 evaluation conditions, with 80 *pairs* (160 individual episode decisions) per writer/probe/seed. Across 3 seeds × 3 writer origins × 3 probe kinds, **27 probe checkpoints / 4,320 evaluated model-episode decisions**.
- Report full pre-repair accuracy, opposite-past both-correct rate, state-swap accuracy, exact zero-state accuracy, historical-role breakdown, H=512 stress, Euclidean baseline, and state pair L2. Confidence intervals must use independent writer seeds, not individual partner episodes.
- The untrained writer probe is an essential control: a random recurrence plus powerful downstream supervision can in principle extract information. Improvement is not automatically caused by *trained memory* unless it outperforms that control.
- **Probe trainability is not proof of recoverable general memory**: if training and testing share synthetic generator, interpolation and shortcut concerns remain. No naturalistic environment, no multi-domain real-world holdout, no future latent prediction are included.

## Reproducibility and fail-closed validation

GitHub Actions explicitly downloads the archived prior ZIP via read-only GitHub Actions API and validates the full artifact hash (no mutable latest-run aliases). For every seed/origin, SHA-verify frozen parent checkpoint bytes, verify model state hash at start/end, save 27 probe weights and raw held-out predictions, original train/eval visible fingerprints, reset and swap predictions, oracle Euclidean control, and source hashes.

Validator reloads all 27 new probe checkpoints and all parent writer checkpoints, regenerates evaluation states and rows from seeds, checks hash/integrity, data cardinality, no train/test overlap, and all prediction metrics. **Evaluation replay does not reproduce probe optimization and does not establish independent scientific replication.** If source archives expire or a checkpoint cannot be audited, fail the experiment, do not silently initialize replacement weights.

## Interpretation gates

- If frozen trained writer + flexible state probe exceeds 50% but original DEV-010 readouts stayed near 50%, that supports **decodable historical information** under the probe's extra supervised training and capacity, not a CED advantage.
- If flexible trained-writer probes remain at 50% while full-history Euclidean is high, a **finite-memory information/recoverability bottleneck is plausible**, but cannot be proven absent without stronger probes and longer training.
- If the random/untrained writer performs equally well, learned writer specialization is **not demonstrated**.
- If performance collapses at H=512 or D=32, report interference and long-horizon limitation.
- If zero-state exceeds 50% with paired evaluation, reject as data/label leakage or faulty control.
- Do not change DEV-010 frozen negative results retrospectively. The relevant next work (if needed) is a **writer intervention** with protected writes, schema-aware indexing or a modern gated-delta recurrence at matched state capacity and compute, not indefinite scaling of a new reader.
