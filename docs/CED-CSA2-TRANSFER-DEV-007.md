# CED–CSA2 transfer study for TAC-OSM (DEV-007)

> **Status: DESIGN / DEVELOPMENT ONLY. NOT AN IMPLEMENTED MODEL, TRAINED RESULT, SCIENTIFIC PREREGISTRATION, OR SOTA CLAIM.**
>
> Drafted 2026-10-12 after reading DeepSeek-V4.1-Flash (arXiv:2609.19969v1), YOCO (2405.05254), IndexCache (2603.12201), and TAC-OSM DEV-003–DEV-006 evidence. Branch is stacked on DEV-006 PR #162 for review. Do not merge out of dependency order. Freeze a revised protocol and external evaluation splits *before* training; remote publication here does not retroactively preregister earlier observations.

## 1. Scientific question

Can a **causal encoder + reused global KV predictive reader** reduce the cost of repeated historical state use in TAC-OSM **without degrading** delayed retention, exact retrieval, action-conditioned future prediction, and downstream decision quality?

This is separate from:
1. Whether learning a useful finite persistent memory is solved (it is not).
2. Whether a sparse index gives **sublinear query work** (DeepSeek's first hierarchical selection still scans the available context).
3. Whether predicting a handcrafted latent is evidence of a real world model (it is not).
4. Whether inference KV caches are persistent, writable, learned episodic memory (they are not).

Maintain TAC-OSM's actual transition loop: state -> address/retrieve R -> route/execute -> observe/verify -> staged write. Keep the learned finite-state pathway intact as a controlled input or parallel diagnostic, not silently replace it with an unbounded full-history cache.

## 2. Verified DeepSeek mechanism and transfer boundaries

- **CED**: 20 causal encoder + 20 decoder layers in the published 40-layer model. The decoder global KV entries are projected from the final encoder state with layer-dependent weights. Most prompt positions need not traverse the decoder during prefill; decoder SWA still requires a bounded replay. The published complexity is approximately O(NL/2 + WL/2), not a guarantee about arbitrary predictive models.
- **CSA2**: layer modes Full (create global KV and index), Reindex (reuse KV, select new indices), Reuse (reuse KV and indices). Every mode retains its own query and local-window processing.
- **Hierarchical sparse indexer**: first Full-mode layer scans all visible positions and builds a candidate pool; later Reindex layers restrict themselves to that pool. Its savings are on *repeat indexing*, **not** on the first scan.
- **890 bytes per token** is DeepSeek's own global FP4 KV-cache accounting, not a capacity guarantee for TAC-OSM. 1/8 persistent cache is a different claim that also depends on SWA replay and storage policy.
- **Engram** is tokenizer/n-gram conditional table lookup, not arbitrary online episodic learning. DSpark and Mega-mHC are lower priority for this experiment.

References: https://arxiv.org/html/2609.19969 ; https://arxiv.org/abs/2405.05254 ; https://arxiv.org/abs/2603.12201 ; https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash

## 3. Existing measurements and decision to delay predictive-objective scaling

DEV-003: 54.86% learned historical decisions vs 86.81% full-history Euclidean; learned finite-state memory uses 768 scalars. DEV-004: delayed verified carry falls to ~50% after 32 competing writes; retention and interference are open. DEV-005: learned protection 56.94% vs always-update 57.52%, so no advantage established.

DEV-006 [run 38288728954, full artifact 11764717412 at SHA a4c517fea3779d5ed6bb9fadcfc8756ade8d08ce] completed with job success and checkpoint evaluation replay validation. From saved `report.json` (three seeds; 576 episodes per arm per seed):

| Arm | Mean decision exact accuracy | Mean future latent MSE |
|---|---:|---:|
| task_only | 55.73% | 0.6251 |
| future_only | 49.83% | 0.4419 |
| joint | 56.42% | 0.4410 |
| joint_shuffled | 50.23% | 0.6311 |

Task joint minus task_only seed gaps: +2.60, -2.26, +1.74 percentage points; mean **+0.69 pp**; exploratory df=2 95% t-interval approximately **[-5.75,+7.13] pp**. All future MSE means exceed the simple public-only conditional-mean MSE 0.3367. For H=512 task_only 56.08% and joint 55.38%. **Therefore predictive superiority and decision transfer are not established.** Keep the result development-only; replay does not reexecute/reproduce training, and three seeds cannot support a confirmatory inference. Do not upgrade it to positive evidence merely because the workflow succeeded.

Archive: https://github.com/FPC-effortless/TAC-OSM/actions/runs/38288728954 ; DEV-006 design: docs/PREDICTIVE-WORLD-DEV-006.md.

## 4. Minimal scientific implementation, not a 552B replication

Build a small torch module behind a **new development-only adapter**; avoid editing the frozen DEV-003–DEV-006 implementations, evidence, training scripts, and reference checkpoints.

Inputs: ordered visible observations `o[0:H]`, visible action history `a[0:H-1]`, query `q_t`, current observation `o_t`, and separate finite persistent memory `M_t`. Strict causal masks, no future observation/target, no evaluator-only past label/answer, no target slot, no role ID, no gold retrieval identity.

Encoder `E_theta` processes causal prefix states `h_i = E(o_{<=i},a_{<i})`. Decoder `D_phi` predicts decisions / future latents and reads layer-specific `(K_l,V_l)=P_l(h_{0:H})`. It may compute local-window states from its own stream; output queries must not access future positions. Do not describe this as vanilla T5 bidirectional cross-attention.

Optionally produce a bounded persistent summary `M_t = U(M_{t-1},o_t,a_{t-1},verification_t)` (existing memory writer, unchanged in the first pass), and a selected set `R_t = Index(q_t, M_t or cached historical h)`. Do not conflate external/episodic persistence with inference KV cache. Use separate instrumentation for each.

### Primary factorial design (four architecture arms)

Use identical visible examples, data-order fingerprints, source encoder/decoder width, total depth, optimizer and loss. Tune widths if needed to compare *parameter-matched* arms; separately compare *training/inference FLOP-matched* arms and explicitly state which comparison is being made. Training/eval task uses **task-only decision loss first**, to avoid confounding CED with DEV-006's failed auxiliary objective. Preserve the same finite-memory writer and identical capacity unless the memory factor is explicitly ablated.

- A. **Dense standard Transformer**: four causal self-attention blocks, per-layer global KV.
- B. **CED dense reader**: two causal encoder blocks + two decoder blocks with independently projected encoder global KV, all history positions accessible; shared-write semantics but no sparse selection.
- C. **CED shared-KV**: as B, two decoder layers reuse a single global KV representation; each decoder layer has its own queries and feedforward transforms. No hierarchical indexing.
- D. **CED + CSA2-inspired selection**: as C with one Full scoring layer and a Reuse layer, bounded top-K reading. Add Reindex as a separate arm if Full vs Reuse quality is non-inferior.

Arm D must not be sold as sublinear, as its first Full pass scans O(H) positions. A must not be compared to B-D at inconsistent active params or different memory capacities without declaring the mismatch. Task-only, deterministic frozen synthetic downstream task is initial mechanism isolation, **not** a general world model benchmark.

### Second-stage diagnostic (run only after the first evaluation is audited)

Replace the handcrafted DEV-006 latent with **independently frozen future-observation encoder targets** or a transparent action-conditioned dynamical simulator with held-out transition families. Train a matched task-only/joint/shuffled-future factor across the winning *and* baseline architectures, with the future predictor seeing only available past, current observation and selected action. Evaluate both counterfactual action branches on the same history, not action parity confounded with task role.

Report future prediction relative to (a) current-only conditional mean, (b) simple classical transition baseline, and (c) no-memory state. An accurate auxiliary prediction without improved downstream decisions and memory retention must remain diagnostic.

## 5. Evaluation validity and stress tests

- Keep existing opposite-past matched pairs, both retention and acquisition roles, balanced binary actions, same-present/different-past counterfactuals, near-address competitors, and strictly disjoint train/eval sample fingerprints.
- Scale H across 32, 128, 512, 2048 and 4096 (the latter two only when implementation supports it). Vary competing writes D across 0, 4, 16, 32, 64; vary noise, alias density, distractor count and relevant-set size. Report actual observed coverage for every condition; never quietly omit hard conditions.
- Include controls: reset memory; shuffled/wrong history; no-write; oracle address (diagnostic only); Euclidean and cosine full-history retrieval with **full cost**; cache hit vs miss; action swap; candidate-pool miss; stale KV invalidation after write; first-token and late-token retrieval.
- Verify no answer can be inferred from action parity, role, position, exact-key identity or target leakage. All pair members must share present observation and action when testing past causality. Assert causal truncation (modifying future inputs does not change predictions of earlier times).
- Train with >=5 genuinely different initialization seeds and fixed examples/steps per arm; use independent evaluation streams. Reserve untouched holdout domains for later prospective work. Report all seeds, matched episode-level rows, checkpoint hashes and per-condition scores. Confidence intervals must be over independent training seeds, **not** treating 1000 evaluation episodes as 1000 independent trained models.
- Separate **accuracy**, **target retrieval recall@K**, **state sensitivity**, **long-delay retention**, **false memory writes**, **future-latent MSE**, **action-counterfactual response** and **real decision transfer**.
- Record actual parameter counts, active operator calls, measured FLOPs/MACs, KV bytes and tensors, index construction, query scoring positions, pool rescoring positions, local replay token counts, wall clock under controlled hardware, peak allocated memory, history reads/writes, and amortized cost at declared query-to-build ratios.
- Reject invalid evidence on checkpoint/source mismatch, missing examples/cells, leakage, wrong causal masks, unbalanced actions, untrained pathways, misleading work counters, or replay mismatch. CI green indicates software verification, not a discovered capability.

## 6. Cost hypotheses and failure criteria

Let H be history size, K selected items, B candidate-pool size, L depth, W local window, Q number of queries since index build.

- Conventional dense attention cache (illustrative): O(L H d_KV) global KV bytes and O(L H) scoring per generated query.
- Shared decoder KV: O(G H d_KV) for G unique producing groups, where G < L (other caches/replay must be added separately).
- CSA2-first-index + shared pool: O(H) for first per-query indexing + O(B) for each reindex layer + O(K) sparse reads per attention layer. End-to-end per query is **still O(H)** unless the first stage uses an actually sublinear index.
- Query-time indexed alternative (separate DEV experiment): evaluate an independently maintained ANN/hash/tree index, charge **build, update, storage, collision, recall and maintenance**; report amortized cost `C_build/Q + C_update + C_query + C_execute + C_verify`. No sublinear claim from a fixed top-K decoder alone.

Predeclare the engineering screening target *before new outcomes are observed*: a putative efficiency-positive arm must maintain held-out accuracy within 2 percentage points of a matched comparator **and** reduce measured prefill work by >=20% or global KV bytes by >=25%. These are **provisional design targets** and should be refined/frozen before measurement, not retrospectively fitted. Show intervals and all tradeoffs even if targets are missed. No criterion automatically establishes world-model capacity or asymptotic improvement.

Stop/negative interpretations:
1. If latent state compression reduces retrieval recall especially at H=512+, label **information loss**.
2. If the first index scan dominates, report **constant-factor reuse only**.
3. If positive task gaps occur only for unbalanced action/role strata, reject **confounding**.
4. If decision performance improves with unrelated future labels equally, reject causal auxiliary-learning claim.
5. If memory interference remains at chance, do not advertise CED as a persistence fix.
6. If wall-time or FLOPs regress, report cache-byte savings separately instead of claiming overall efficiency.

## 7. Proposed implementation order / deliverables

1. **Design audit (this document):** validate citation-to-mechanism mapping and current DEV-006 artifact outcome. No model implementation claimed.
2. **DEV-007A code:** reference dense causal attention and CED projection with explicit masks, tests for equal causal prefixes, stale-cache invalidation, and numerical equality of reused readouts where expected.
3. **DEV-007B attention reuse:** shared-KV and Full/Reuse/Reindex selectors; assert all work counters match actual operations and first-layer scan is charged.
4. **DEV-007C data/experiment harness:** fixed-budget matched training and stress evaluation, raw/checkpoint replay validation, test-failure injection, CI smoke, explicitly opt-in full development run.
5. **Results audit:** compare all arms and baselines, publish negatives, then make a *separate* decision whether to expand to frozen future-state targets and compute-outcome prediction.

## 8. Claim ledger

| Claim | Present status |
|---|---|
| DeepSeek CED reduces prefill on DeepSeek V4.1-Flash | **DeepSeek-reported**, supported by technical report |
| CSA2 shrinks DeepSeek's global KV footprint | **DeepSeek-reported**, model-specific |
| Such reuse helps TAC-OSM decision quality | **UNTESTED** |
| Such reuse fixes TAC-OSM finite memory interference | **UNTESTED** |
| Joint DEV-006 improves learned decision quality | **NOT ESTABLISHED** by three development seeds |
| DEV-006 predicts better than the conditional-mean future baseline | **NO** on mean MSE in saved artifact |
| CED + hierarchical index yields TAC-OSM sublinear total compute | **NOT ESTABLISHED**; DeepSeek first-stage scan remains O(H) |
| TAC-OSM exhibits general action-conditioned world modeling | **NOT ESTABLISHED** |

The next merge, if any, should carry engineering instruments with explicit negative/unknown claim status. Do not change earlier development reports or promote this design to a confirmed result.
