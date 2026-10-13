# DEV-010 — DeepSeek-inspired CED reading over TAC-OSM's finite learned PLM

**Status: DEVELOPMENT-ONLY / NOT CONFIRMATORY.** The design and implementation do not establish a state-of-the-art world model, better reasoning, robust persistence, generalization or sublinear computation. Preserve earlier negative/mixed DEV-003–009 evidence unchanged.

## What changes compared with DEV-009

DEV-008/009 encoded a full list of visible **candidate descriptors** and routed inside the hardened loop. That proved integration contracts, but did **not** exercise TAC-OSM's **trained finite recurrent memory**.

DEV-010 reads the **actual 768-scalar recurrent state** from `ProtectionModel(mode='always_update')` (four experts × eight value channels × 24 key channels). It inherits:
- Learned `DeltaMemory` keys, writing, erasure and decay at each observation.
- Noise and competing-write tasks built with `make_pair` and counterfactual past-bit flips.
- Existing conditional structural executor, selector and halt logic.
- Original binary task answer and CE objective.

The reader **only** replaces the expert-aggregation pathway: a permutation-equivariant expert-set attention encoder projects one global KV representation and two query-only attention decoder reads. This is **CED-inspired**, not the published 20+20-layer DeepSeek CED, not a temporal encoder over all raw history, and not a genuine retrieval index. The recurrent memory writer already enforces temporal causal ordering. Memory experts are slots and must not receive arbitrary chronological positional encodings.

**Integration boundaries:** DEV-010 works inside TAC-OSM's learned finite-memory/conditional-compute development model; it does **not** demonstrate a full system joining that trainable PLM to `HardenedLoop`'s repair and verifier. DEV-008/009 tested the hardened-loop contract separately.

## Exact comparisons and why they are diagnostic

Two models are trained per seed:
- `native`: unchanged `ProtectionModel(mode='always_update')` expert readout, with its top-k selector.
- `ced`: identical initially copied weights for all common PLM writer, query, compute and answer modules; additional CED reader trained with the **same** CE task examples.

The reader architecture has extra parameters, so native-vs-CED **is not parameter-, active-FLOP-, or throughput-matched**. Its strength is isolating whether the *existing finite PLM substrate* accepts a CED readout that gradients can train.

After CED is trained ONCE per seed using its **dense** mode, evaluate **the same checkpoint** twice: `ced_dense` and `ced_sparse`, with second attention layer restricted to top 2 of 4 expert slots in sparse. No supervised expert oracle, no gold slot index, and no masked candidate-class logits. Both dense/sparse get the same first-stage full scan, same storage, same weights and same evaluation episodes. Sparse routing cannot be credited with a training advantage.

The expert set attention itself is **not causal in slot order**; it is permutation-equivariant, which is correct because experts have no time ordering. The recurrent state remains temporally causal. This differs from DeepSeek's causal token encoder and must be described accurately.

## Development protocol

Training: three seeds 43, 59, 71, paired native/CED optimizer, 72 updates/seed/model. Each step includes 2 opposite-past paired cases (4 episodes), with balanced retention/acquisition and alternating history H=32/128, competing writes D=4/16, near-interference fraction 0/1, noise 0.08/0.25. The evaluator-only answer never enters `predict` or memory update; the visible observations, addresses, query and current bit are the only forward inputs.

Independent evaluation seed pairs 241/257/269. Each of 32 base conditions H=32/128, D=4/16, near 0/1, noise 0.08/0.25, role retention/acquisition is paired; eight extra long-history H=512/D=32 cases test stress. There are **40 opposite-past pairs = 80 episodes per inference mode per seed**; three inference modes across three seeds means 720 measured model-episode evaluations. These are *exploratory samples*, not a prospective untouched benchmark. Pair partners share current inputs but differ in one historical bit. All three prediction modes use the exact same evaluation pairs.

Primary endpoints: **pre-repair exact task decisions**, role-conditional accuracy, and 512-history stress. Report every seed and reset-memory controls. Secondary: memory-pair L2 contrast, paired both-correct rate, recurrent memory write events, shared KV scalars, logical encoder/decoder score pairs, and native-vs-CED parameters. All evaluations are diagnostic, not real-environment future dynamics.

**Control validity:** train/eval input fingerprints disjoint; model writer and CED reader receive nonzero gradients; source SHA-256/checkpoint hashes and raw per-episode results saved; replay reconstructs all decisions from checkpoints. Replay does **not** rerun the original SGD optimizer or prove scientific independence. Do not treat 720 episodes as 720 independent training seeds.

## Correct cost interpretation

Each episode stores only 4×8×24 = 768 recurrent state scalars, regardless of historical H. But every observed event is still **processed by the learned writer**, an O(H) ingest cost. CED encoding of four expert slots adds compute. Shared global KV contains `2 × B × E × dim` scalars at current read and is not the finite PLM state. Sparse restricts only the **second query decoder read** from four experts to two; the expert encoder and first global read still process all four. This cannot support an end-to-end sublinear computational claim.

## Interpretation rules

1. If CED improves accuracy only with more parameters, **no efficiency conclusion**.
2. If sparse preserves dense accuracy but reduces second decoder score positions, report **small logical attention reduction**, not hardware speedup.
3. If both stay near chance, the underlying 768-scalar learned persistence likely remains a bottleneck; test readout vs memory erasure with frozen-read probes before adding architectural complexity.
4. If accuracy rises but reset-memory produces comparable accuracy, reject a persistence explanation.
5. If apparent improvement appears only at one seed, report per-seed variation; no positive scientific conclusion.
6. Any future integration into full verifier/repair loop requires separate end-to-end tests where the learned memory actually supplies the pre-action router.

The problem under study is **whether a query-only CED reader helps an existing learned persistent memory**, not whether an LLM can replace TAC-OSM.

External precedents: DeepSeek CED https://arxiv.org/abs/2609.19969; YOCO https://arxiv.org/abs/2405.05254. These are precedents, not proof of transfer.
