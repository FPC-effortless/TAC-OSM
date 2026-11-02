# COLLECT011: phase-address memory in the actual TAC language-model graph

Status: **engineering integration tested; structured component repaired; language-model training blocked**. No optimizer updates, new fits, sealed TEST access, architecture-quality ranking or merge is authorized by these diagnostics.

This package vendors a hash-pinned, minimal executable closure from the user's TACOPT114 source artifact. It uses the actual four-layer `ResearchTAC110` graph, with SSA enabled and mHC disabled. Native fast/slow associative banks, rotating field, perspective mixer, FFN, byte embedding/output and mid-stack SSA remain. The experimental arms replace the local token-attention lane, rather than substituting a small unrelated SSM for TAC.

## What changed

`collect011/memory.py` separates address, value and admission projections. Shared content/role phase addresses feed a read-before-write bounded memory. The normalized inverse-covariance write from COLLECT010 remains differentiable through both carried matrices. The new LM frontend is **untrained**, is not the frozen COLLECT010 character encoder and has no validated semantic key extractor. Identical feature vectors share an address; semantically equivalent language mentions are not guaranteed to do so.

The registered arms are:

| Arm | Local lane | FFN width | Learned/deployed parameters |
|---|---|---:|---:|
| attention | Original causal RoPE attention, window 320 | 64 | 208,227 |
| delta | Wide phase address, ordinary delta write | 85 | 208,231 |
| phase_p | Wide phase address, preconditioned write | 85 | 208,231 |
| hybrid_p | Four phase banks, soft routing | 83 | 208,387 |
| hybrid_hard_p | Four phase banks, discrete forward routing and straight-through gradient | 83 | 208,387 |
| phase_p_support | Wide preconditioned memory, smooth support gate on its residual | 85 | 208,231 |

The maximum parameter gap is approximately 0.077%. No inactive attention weights or padding parameters remain in replacement arms. Both hybrid implementations currently compute dense projections and matrix updates; discrete routing does not establish a sparse GPU speedup. Hard routing uses a biased straight-through surrogate, not the mathematical derivative of argmax; finite/chunk-consistent gradients do not validate that estimator for training. The support-gated LM arm uses precision-derived smooth evidence, not the structured-record Bloom filter below.

## A concrete diagnosis and repair

The frozen weaker COLLECT010 seed made the same two errors in FP32 and FP64 on new 11B histories. Both latest values decoded correctly before storage; their commit probabilities were about 0.258 and 0.496, below the hard 0.5 threshold. Their old committed values exactly matched the erroneous answers. These observed failures were missed writes, rather than numerical precision or absent-key rejection failures. This does not prove that every earlier failure had the same cause.

Two explicit structured-record repairs were tested without refitting:

1. **Validated commit event:** accept a write only when the observed suffix is exactly `[A-H]{3}:[a-p];`. It uses current/past characters, not key identities, query answers or future tokens. It replaces the learned commit decision with known grammar structure.
2. **Independent support lane:** add the observed key bytes to a 4,096-bit, four-hash membership filter at a validated commit. A query that fails membership returns NONE; a membership hit uses the memory decoder without the old geometric rejection threshold. The filter adds 512 bytes per owner. It is probabilistic, can saturate, and supports reset but not deletion/eviction. It does not retrieve values or discover entities in natural language.

The first repair recovered all 4,224 present queries and all 192 maps on the 13B replay; one of 3,072 absent queries was still accepted. A second, predeclared 15B replay tested all three modes on the same histories. Validated commit plus the separate support lane recovered all 4,224 present queries, rejected all 3,072 absent queries, and recalled all 192 maps perfectly across three frozen seeds and four loads. This is a grammar-specific synthetic component result. It is not a trained LM result, an unseen-grammar result, or a production reliability guarantee.

## Engineering evidence

The release checks cover the source-matched native attention control; all six full TAC graphs; finite active parameter gradients on original TRAIN bytes; attached-state chunk/token/gradient parity; future-token poisoning; owner isolation; serialization; zero-write freezing; mixed/invalid precision rejection; SPD state validation; normalized-write gradients; causal grammar validation; and membership saturation/reset behavior. The generic probe adapter exposes fast/slow banks, rotating field, SSA and phase matrices. Variable native KV caches are covered separately by full-graph tests; zeroing a precision matrix is an invalid-state intervention and remains INCONCLUSIVE in the generic memory probe.

Long-generation checks use **fresh untrained weights**: 512 sampled tokens at temperatures 0.8 and 1.2 check finite logits/state and bounded carried storage. They do not measure coherent text, perplexity, retrieval learning or trained quality. Constant carried-state size also does not imply constant BPTT activation memory.

CPU forward/backward timings and state bytes are recorded separately. CPU results do not establish T4 efficiency. The preconditioned operator adds 64 KiB of precision state across four wide layers, or 16 KiB across four hybrid layers, in FP32. This differs from COLLECT010's single 16 KiB precision matrix. Actual total state includes all remaining native TAC and SSA fields.

## Reproduce

Python 3.11 and Torch 2.15.0+cpu were used. Run commands from this directory. Obtain the original `TACOPT114_Code_and_GPU_Evidence.zip` and `COLLECT010_reproducibility_bundle.zip` supplied with the preceding work. The source identities are recorded in `SOURCE_PROVENANCE.json`.

```bash
python -m collect011.restore_data /path/to/TACOPT114_Code_and_GPU_Evidence.zip
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. python -m collect011.run_release_checks
python -m collect011.c10_precision_probe --prior-root /path/to/COLLECT010
python -m collect011.c10_failure_diagnosis --prior-root /path/to/COLLECT010
python -m collect011.commit_repair_probe --prior-root /path/to/COLLECT010
python -m collect011.commit_repair_probe --prior-root /path/to/COLLECT010 --protocol SUPPORT_REPAIR_PROTOCOL.json --output SUPPORT_REPAIR
```

The data restorer parses the frozen launcher as literals, never executes it, verifies the embedded payload SHA and split hashes, and materializes TRAIN/DEV only. Original corpus bytes, fitted checkpoints and sealed TEST are excluded from this change. The old component checkpoints remain in their original artifact and are loaded with `weights_only=True`.

## Training and mathematical status

`FAIR_COMPARISON.DRAFT.json` is an unqualified plan, not a passed preflight. It declares an approximately 208K target under the v2 policy snapshot used by TAC110. The default repository branch still carries the older 100K contract; these policy versions must be resolved by the canonical integration workflow before fitting. The v2 preflight also fails the unresolved source-authority requirement, and the canonical signed verifier independently rejects missing evidence. Actual full-update FLOP budgets/allocation and complete source-bound PNDS verification are open. No trusted key or attestation is generated here.

The independent Python scalar reference checks forward/backward agreement of the transferred operator. `math/PhaseWrite.lean` formulates that operator and clock bookkeeping only; Lean compilation, the phase/frontend/native full graph, IEEE semantics and AD proofs are **OPEN**. It is not a full source-equivalence certificate or a completed reciprocal PNDS registration.

The recommended next fit must establish semantic query/write alignment and write admission from causal text, using identical data order and budgets across the six arms. The parser/filter repair supplies an explicit structured interface when record boundaries exist; it does not remove the need to learn boundaries for ordinary language. A fused read/write/precision kernel and genuine sparse hybrid execution are performance hypotheses requiring separate GPU measurements.
