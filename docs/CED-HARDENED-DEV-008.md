# DEV-008 — CED candidate-reader in the TAC-OSM hardened loop

**Status: DEVELOPMENT / MECHANISM + SYNTHETIC INTEGRATION ONLY.** This is **not** a DeepSeek reproduction, validated learned PLM, CASM result, world model, sublinear retrieval system, or independent pre-registered study. The branch is stacked on DEV-007 PR #163, which itself depends on earlier research PRs. Do not merge out of dependency order.

## 1. Why this exists

DEV-007's earlier CED training arms were **standalone synthetic retrieval models**: they did not run inside TAC-OSM's persistent state, retained-set router, verifier and delayed write. DEV-008 tests this actual integration boundary before scaling architecture claims.

The insertion is exactly at the hardened loop's already existing callback:

    router(public_query, temporal_state, retained_candidates)
         -> RoutingDecision(selected_local_index, scores)

No changes are made to `HardenedLoop`, `TemporalPersistentState`, the candidate address index, synthetic `RelationExecutor`, verifier, repair, or previous checkpoints/reports.

## 2. Architecture and ownership

- The `QueryOnlyCED` causal encoder processes a **set presented as an ordered sequence of retained candidate descriptors**. It does **not** encode the entire environmental observation history or learn the finite PLM state transition. Its causal mask has prefix invariance, but candidate sequence ordering is a possible confound and is **not** a time series.
- The decoder operates on just one publicly available query embedding. It does not replay retained candidate tokens through the decoder. It uses global K/V projected from the encoder final states, not from intermediate decoder token representations.
- `HardenedCEDRouter` accesses the persistent state only through **one** `state.read(query)`. If a staged write has not become available, the router cannot see it. It may use the binary reference carried visibly in the query text for non-state tasks. If neither reference is available, it raises; no fallback to private oracle metadata is allowed.
- The candidates passed to CED are only those already retained by TAC-OSM's address index. CED **cannot** reintroduce excluded candidates by looking at `GeneratedTask` or `task.reference_bits`.
- The router does not select outcomes, repair decisions, state writes, executor nodes or hidden correct actions. Those remain under TAC-OSM's original call graph.
- `dense`: two independent encoder-projected global KV sets; `shared`: a single KV set reused across both decoder layers; `sparse`: a shared KV with full first-layer scoring, then reused top-k indices for the second global read and final routing score. This is CSA2-*inspired*; the first scan remains linear in the number of retained candidates.
- Logical work accounting includes 2 encoder full attention matrices, first global score pass, second selected score pass, and final score pass; it separately records decoder-global KV scalar storage. These are **not** hardware FLOPs or latency.

## 3. Existing mechanism evidence

On 2026-10-13, CI job from GitHub PR #164 at commit e22888dd completed **9 focused tests passed** and collected **949 total tests**. It checked causal-prefix invariance, initial weight identity for shared/sparse modes, real backprop gradients, delayed persistence boundaries, no hidden state access, full hardened-loop execution and verification, retained-set indexing, score-pair and KV-count arithmetic, and explicit errors on missing public context.

This proves only that the **mechanism contract and integration tests work**; not learned predictive capability or quality. Synthetic executor and environment remain controlled references; tests are not a comprehensive safety/correctness proof for all workloads.

## 4. Development-trained pilot design

`scripts/run_ced_hardened_dev.py` trains shared/sparse modes with **matched initialization per seed** and the same procedurally generated unique-equality samples. Supervised target indices are passed to the training loss only. At inference the model receives only public reference bits, marked positions, and retained candidate descriptors.

- Training: three seeds 43, 59, 71; 120 AdamW steps per seed and mode; candidate count 8. **This budget is exploratory and not prospectively registered before exploratory results; do not present a statistical confirmation.**
- Tests: untrained vs trained results need separate future work; this first pilot evaluates learned shared/sparse on candidate counts 8 and 32, each at delay 0 and 4, 24 distinct evaluator episodes per cell.
- Every evaluated case instantiates the **real** `TemporalBenchmark`, `HardenedLoop`, `RelationExecutor` and verifier, with repair disabled and learn-on-success disabled solely to isolate pre-repair routing. Delay=4 explicitly traverses all intervening temporal decision boundaries. Model training itself does not occur end-to-end through the loop's verification or state-write controller.
- Primary endpoint is **pre-repair exact decision accuracy**. Report by independent seed, size and delay. Secondary endpoints: logical encoder/global attention scoring, cache scalar counts, executed structures and verifier calls, and memory source (public query vs delayed state read).
- The unique-equality synthetic task is **solvable by an exact nonlearned signature comparison** from public bits; do not claim real reasoning/generalization or SOTA from good accuracy. A classical signature baseline is needed before any capability claim.
- Save all per-episode predictions and six state-dict checkpoints in a GitHub Actions artifact. Success of a workflow proves only completion of the code path; saved outcomes must be inspected and verified before interpreting results.

## 5. Stop gates and next step

1. If training results match a simple deterministic equality oracle, do not claim new capability; the generator is intentionally easy.
2. If candidate count 32 degrades despite success at 8, report **longer-set generalization failure**.
3. If sparse is slower or less accurate while using fewer KV scalars, report savings and regressions separately.
4. Once candidate integration is verified, the scientifically relevant next stage is a CED query reader over **learned finite 768-scalar memory** from DEV-003–006 and real noisy-address/competing-write benchmarks, with a matched native non-CED router and pre-repair decisions. Only there can we test whether CED helps TAC-OSM's actual persistence bottleneck.
5. No PR merge until the dependency chain is reviewed. No positive architecture claim from green CI or from the current deterministic memory container.

Sources: DeepSeek-V4.1-Flash https://arxiv.org/abs/2609.19969 ; YOCO https://arxiv.org/abs/2405.05254 ; TAC-OSM repo docs CED-CSA2-TRANSFER-DEV-007.md.
