# SENSE-TOK-001 — sensory / relevance / structural tokenizer comparison

**Status:** Source-only, no-fit mechanism prototype. **Not an authorized model-training or capability comparison.** All seven implementations are deterministic pre-model variants with zero learned parameters. This is an engineering exercise; no synthetic test can promote an architecture. No real FIT/DEV/TEST dataset or publisher permission has been provided. Existing TAC-OSM negative evidence is unchanged.

## Explicit hypotheses and falsifiers

| ID | Mechanism | Prediction to check after qualifying a real-data study | Distinct falsifier |
| --- | --- | --- | --- |
| T0 | `fixed` byte patches | Practical reference | Any gain over it must include the cost of adaptive processing |
| T1 | `entropy` causal byte surprise | More resolution at unpredictable transitions | If prefix inference or end-to-end utility worsens, reject |
| T2 | `sensory` local class contrast / change | Distinct local transitions are useful boundaries | If disruptive nuisance changes dominate, reject |
| T3 | `hierarchy` grouping sensory leaves | Reusable multi-scale representation helps tasks | Compare with T2 at matched total nodes/compute |
| T4 | `relevance` visible query/byte-gram relevance | Query-dependent resolution supports retrieval | Poison the query; reject if gold/label channels are required |
| T5 | `structure` observed sequential/repeat/type graph | Explicit structural links help downstream models | Remove/shuffle edges and compare without hidden true edges |
| T6 | `hybrid` entropy + sensory + relevance + structural edges | Complementary mechanisms produce nonadditive utility | Leave-one-out ablations against T1–T5; reject extra costs if no gain |

**Important boundaries:** T1/T2/T0 cut decisions only from observed prefix bytes; T3/T4/T5/T6 are offline feature construction and are **not safe drop-in autoregressive LM tokenizers** without a separately reviewed streaming/prediction-time contract. The hybrid does **not** currently include a trained CDL teacher or execute the CASM-S model. Its `relevance` score is a transparent query-gram heuristic and its graph is an observed-edge scaffold, not inferred hidden executable semantics.

**Biological analogy:** local contrast/change and hierarchical feature assembly, *not* a detailed fly retina or optic-lobe simulation. The real fly retina and optic lobe are different anatomical/circuit processes. No biological performance claim is implied.

## Typed input/output and invariants

- Input: observed bytes `x[0:n]`; T4/T6 additionally take an explicitly supplied predecision query `q`. No target, future, reward, teacher gold or oracle graph is accepted.
- Output: ordered `Patch(start,end,payload,salience)` partition, optional lower-level leaf partition, bounded forward-only `Relation(src,dst,kind)` set, a source hash, and declared scan work.
- Every patch is a half-open interval `[start,end)` with its actual input byte content. For all variants, `concat(patch.payload) == x` and `sha256(concat(...)) == sha256(x)`. The hierarchy's leaf partition has the same identity.
- Edges are public observational links only (next/equal-bytes/same-byte-class), **not** CASM candidate gates or true edges. They use observed graph state, not evaluator truth.
- Time: direct per-byte segmentation requires only prefix observations. Offline refinement is qualified **only** for inputs fully observed before a downstream decision.
- Objective: none fitted; no gradient paths or parameter counts. Training and full formal comparison remain blocked.

## Source-reference mathematics (exact code-level operators)

Let `x∈{0,…,255}^n`, and a partition `P={(s_j,e_j)}` satisfy `s_0=0`, `e_j=s_{j+1}`, `e_{m-1}=n`, and `e_j>s_j`. Each patch stores `x[s_j:e_j]` exactly.

- T0: emit patch boundary after every `k` observed bytes (and at `n`).
- T1: with prefix-window multiset `C_i` and length `h_i`, surprise `u_i = -log2((C_i(x_i)+1)/(h_i+256))`. Emit after position `i` when `length≥max_patch` or (`length≥min_patch` and `u_i≥6.4`). The current byte is appended to the history **after** calculating `u_i`.
- T2: `v_i = 2·[class(x_i)≠class(x_{i−1})]+[i≥2 ∧ x_i≠x_{i−1}=x_{i−2}]`; emit at max size or when min size reached and `v_i≥2`.
- T3: greedily group whole T2 leaves without crossing the `2·max_patch` top-level budget (except an already oversized leaf).
- T4: compute the observable query bigram set `G(q)`; patch match indicator `r_j = [G(x[s_j:e_j])∩G(q)≠∅]` (single-byte query fallback). Split relevant T2 leaves into 2-byte subpatches; merge nonrelevant leaves subject to a max patch size.
- T5: for patch `j`, add `(j−1,j,next)` if `j>0`, plus bounded latest exact-repetition and latest same-byte-class predecessors as applicable. Never add an oracle true edge.
- T6: union T1/T2 segmentation boundaries, apply T4 query refinement, then T5 bounded edge construction; keep union segmentation as a separately counted leaf partition.

The exact executable Python is `src/tac_osm/sensory_tokenizers.py`. This note is **not** a Lean proof, matched full-graph mathematical implementation, gradient-equivalence result, or official PNDS reciprocal registration. PNDS formalization, source-pinned reciprocal math record and Lean compilation are **OPEN** and block mathematical release.

## Experimental validity roadmap — preregister before any fit

1. Establish an independently sourced, licensed, checksum-pinned real FIT/DEV dataset with genuine task objectives and stable entity/time boundaries. Keep TEST sealed. No corpus inclusion by mere script location.
2. Freeze tokenization training procedure, downstream model, evaluation objective, budget, seeds, preprocessing fit boundaries, data exposure and measurement method. Train comparable ~100k parameter models (95k–105k each, within 1% pairwise) under *identical original source examples, optimizer and allocated/consumed compute budgets*. Count **all** learned patch encoders, routers, decoders and graph heads. Do not place unused dummy parameters in simpler controls.
3. Metrics: task-specific real-data out-of-time accuracy/byte-level negative log-likelihood (bits per byte for language), exact task completion where appropriate; normalized quality under matched input information; contextual and OOD robustness; bytes-per-patch; full representation bytes including boundaries, edges, query cache and any raw byte retention; actual prefilling/decoding FLOPs, peak memory, CPU/GPU seconds and energy where instrumented; streaming latency and causal boundary violations.
4. Controls: byte-only, fixed patches, actual BLT-like entropy patcher, actual learned H-Net-style chunker when implementations are available; identity/graph shuffle, query shuffled/wrong, feature-null, equal-resolution, preprocessor-disabled, raw-history retrieval; no positive study claim from this repository's simple heuristic stand-ins.
5. Threat model: train-only preprocessing; duplicated and near-duplicated documents, future-target mutation, query gold leaks, answer-in-ID, cue/label leakage, contamination, cross-entity state, offset/candidate-order, run selection, graph true-edge leakage. Each must report PASS/FAIL/INCONCLUSIVE, not assume zero risk. Use `research_anticheat` and a bespoke `ResearchProbeAdapter` for any fitted model. Independently audit checkpoint, evaluator, goal/reward, writer ACL and provenance.
6. Register a reviewed `research/phone_queue` protocol and run `scripts/phone_lab.py plan`. Do not begin training, spend Kaggle GPU minutes, or open TEST until owner-approved 2xT4 quota and all independent signed prerequisite evidence is present. Preserve all failures.

## Running no-fit mechanical checks

```bash
PYTHONPATH=src python -m pytest -q tests/test_sensory_tokenizers.py
python scripts/diagnose_sensory_tokenizers.py --fixture --query 'motor fault'
python scripts/diagnose_sensory_tokenizers.py --input path/to/authorized_FIT_data --split FIT --source 'publisher / version / licence' --query 'predecision query' --output report.json
```

Outputs deliberately report no accuracy, compression advantage, architectural superiority or legitimate real-world capability score. These are tokenizer shape/invariant diagnostics only.
