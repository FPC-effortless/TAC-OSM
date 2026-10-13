# DEV-014 — Isolate TAC-OSM CASM action-rule learning versus finite-PLM retrieval failure

**Status: DEVELOPMENT / NO SCIENTIFIC CAPABILITY CLAIM.** This follows the archived *negative* DEV-013 result (about 60% learned historical-bit diagnostic recall, 52% native task performance). We must distinguish failure to preserve/retrieve a historical payload from failure to perform the action computation once that payload is supplied.

## Critical methodological warning: privileged teacher-bit arms

This experiment is intentionally diagnostic, **not a proposed production TAC-OSM architecture**. In `oracle_sparse` and `oracle_dense`, the **correct historical bit is given directly to a newly trainable CASM input adapter during BOTH TRAINING AND EVALUATION**. This violates real-world memory-inference constraints by design: the oracle condition is only a ceiling/control asking whether the conditional executor can learn a 2-bit XOR-like action rule when relevant information is explicitly available. **Never report the teacher-bit arm as a successful memory model, OSM system, closed-loop reasoner, or actual TAC-OSM prediction.** It is not comparable to DEV-013's actual native memory predictions.

In `predicted_sparse` and `predicted_dense`, the adapter receives only **frozen DEV-013 `aligned_aux` bit probabilities** extracted from the original 4×8×24=**768-scalar** recurrent memory after public observation ingestion; it does not receive a teacher/gold bit or hidden target index. The original pretrained writer/reader is frozen and source/checkpoint authenticated. These are candidate diagnostic *pipelines*, not replacements for the original native `predict` call. Native DEV-013 remains the real system baseline.

## Precisely what is kept and what is new

Reuse the **actual `ConditionalExecutor` class and weights** plus native `answer` layer from the frozen DEV-013 checkpoint. Add **two linear** 4→32 input projections mapping (two historical-bit probabilities, two one-hot public current-bit indicators) into routed and query activations. Input projections are deliberately linear. A linear reader and final answer cannot represent XOR on all four evenly balanced binary states; useful learned nonlinear action transformation must come from the original nonlinear CASM operators (or be exposed as a failure of the ablation assumptions). The original model's actual sparse operator dispatch and halt behavior remain in the call graph.

A no-CASM ablation skips `ConditionalExecutor` and passes the projected routed state straight to the native answer layer, with *identical trained weights*. This gives a direct internal control for whether operator computation is necessary. It is distinct from the top-4 dense-CASM reference which invokes all four conditional operators.

### Four matched adapter arms

- `oracle_sparse`: oracle historical bit train+test, native top-1 CASM operator dispatch.
- `oracle_dense`: identical oracle control with top-4 CASM dispatch.
- `predicted_sparse`: only frozen finite-memory historical-bit probabilities train+test, top-1 CASM.
- `predicted_dense`: same frozen finite-memory probabilities, top-4 CASM.

Each arm has the **same model parameter count and byte-identical starting weights per seed**, including the same pretrained answer and CASM operator parameters; top-k changes executed operators and gradient dynamics, not tensor shapes. Pretrained DEV-013 memory models are frozen in all arms; no memory writer is adapted. All four train on the same target labels and observed historical streams, with 120 updates of 4 examples per seed; all receive equal target supervision and learning-rate settings. Null control: remove historical bit by replacing its distribution with [0.5,0.5] while keeping public current bit. Because counterfactual pair partners have identical present input but opposite targets, this must give exactly 50% accuracy.

### Provenance and registered endpoints

Pin original [DEV-013 Actions run 38440147571](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38440147571), artifact **11836120110**, zip SHA-256 `880d56e177b78d44961a8fc75e6e0a4151ebe6eeb7a55ce81afaa6c358172d65`. Validate exact archive members, all parent model checkpoint byte SHA-256 and tensor parameter hashes. Training / evaluation use entirely distinct `dev014-train` and `dev014-eval` seeded input namespaces, deliberately distinct from the prior DEV-013 training and eval namespaces. Verify no local split fingerprint overlap; the source prior experiment remains adaptive development rather than a prospectively untouched model-family holdout.

For every arm and seed evaluate two replicas of DEV-012's 40 factorial conditions H=32/128/512, D=4/16/32, near collisions 0/1, noise .08/.25, retention/acquisition = 80 opposite-past pairs = 160 episodes per arm per seed, **1,920 total model episode predictions across 12 adapters**. Save primary diagnostic model task accuracy, non-oracle frozen bit-prediction accuracy, no-CASM ablation accuracy, 50% null-bit control, paired swap-bit accuracy, actual invoked CASM operator calls, role and long-history breakdown.

**Decision gate**:
1. Oracle bit + CASM should learn the clean XOR rule (subject to optimization) and beat its **own no-CASM** ablation. If it fails, investigate task transformation/conditional execution.
2. If oracle + CASM succeeds but predicted-bit CASM remains at approximate bit-reader accuracy, the current bottleneck is memory retrieval/bit representation or poor posterior calibration, **not proof that native TAC-OSM is already successful**.
3. If dense exceeds sparse, inspect operator routes and halt instead of attributing weak decisions solely to memory.
4. If no-CASM matches CASM, **the computation was not needed**, and improvement cannot be credited to conditional structural computation. The linear-only ablation should not solve balanced XOR (upper limit around 75% with strict balanced four-combination test).
5. If predicted-bit beats native DEV-013 only by adding this newly supervised adapter, report that extra supervision and inference compute separately; it is not a parameter- or budget-matched native architectural comparison.
6. Parent PLM stays at 768 recurrent scalars, but the new adapter adds model weights, training compute and an extra retrieval head; no efficiency, sublinear time, SOTA world prediction or end-to-end environment capability claim.

All findings are exploratory adaptive development. The validator reconstructs held-out examples and all saved model decisions/checkpoint hashes; **replaying evaluation does not reproduce training SGD**. Keep parent drafts and negative results unmerged and unchanged.
