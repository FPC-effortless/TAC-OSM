# DEV-011 Results — Weak recoverable historical signal, **no benefit from learned writers**

> **Classification: REPLAY-VALIDATED DEVELOPMENT ONLY; NOT CONFIRMATORY, NOT SOTA, NOT A WORLD MODEL.**
>
> We **do not** infer that trained TAC-OSM writer updates improve retention. We **do not** infer that the learned memory has no recoverable information from a failed probe. Old DEV-010 negative results remain unchanged.

## Provenance / replay

GitHub Actions run [38366010881](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38366010881), artifact `DEV-011-RAW-NOT-CONFIRMATORY` (ID **11797306506**, SHA256 `432a1ef78c227d654d89788b875c8791f0a1bcea1727fdc1990afba154d03386`). The original DEV-010 writer archive (ID 11795975699) was fetched by GitHub Actions with `actions:read` permission, checked against full-archive SHA256 `a15685af62122a876c9d990fd877c060c22e41f2d81d80bd47f2426ab558fc56`, and each checkpoint was verified against the parent report.

Mechanism job: **6 focused tests passed**, current total suite collection **964 tests**; smoke checkpoint and prediction replay passed. Full job: **27 separate trained decoder checkpoints** across three writer seeds, three writer origins, and three probe types. Validation loaded all 27 checkpoints, reloaded/reproduced each frozen parent model, checked source and model hashes, regenerated all public train/eval data and exactly replayed **4,320 individual held-out model-episode predictions**, including swapped-state and zero-memory controls. **Replay is not re-running probe training**, and the data generator/architecture were developed adaptively after seeing DEV-010.

## What was isolated

Frozen, unchanged recurrent memories in the same 768-scalar form (`4 × 8 × 24`): (1) DEV-010 native trained writer; (2) DEV-010 CED-trained writer's PLM component; (3) exactly reproduced *original randomly initialized native writer* (hash-matched to the parent report). All three use the same histories and publicly visible noisy addresses.

For each writer, offline-supervised decoder probes fitted on 128 counterfactual pairs at H=32/128:
- `keyed_linear`: expert read `state @ frozen_keys(query)` + public query/current → linear binary prediction.
- `keyed_mlp`: same 32-dimensional expert read + public present → 64-neuron MLP.
- `state_mlp`: entire 768-scalar recurrent state + public present → 64-neuron MLP, intentionally much larger input, **not a parameter-matched fairness baseline**.

All probes saw exactly the same training minibatch schedule and 240 gradient steps for the same seed. Writers were frozen and never optimized. Held-out evaluation used two replicates of DEV-010's 40 registered conditions: H=32/128/512, D=4/16/32, noise .08/.25, near-address competitor fraction 0/1 and both retention/acquisition. **80 matched opposite-past pairs (160 episode decisions) per probe/writer/seed**, 3 independent writer seeds. Public present/query identical inside each pair, history target bit flipped, so zero-memory must yield 50%.

## Held-out accuracy: all 27 probe fits

Values below are means across three writer seeds (160 evaluated episodes per seed per writer/probe).

| Frozen writer | Keyed linear | Keyed MLP | Full-state MLP |
|---|---:|---:|---:|
| Native writer, trained DEV-010 | 48.96% | **55.62%** | 50.21% |
| CED writer, trained DEV-010 | 49.58% | **55.62%** | 50.00% |
| Native writer, **untrained** DEV-010 initialization | 49.38% | **56.67%** | 50.62% |

**The strongest probe architecture is the query-keyed MLP for every writer.** This is *limited evidence of decodable historical information in these fixed-state representations*, because a counterpart with the same present inputs can be classified differently. But the **untrained** memory writer is the descriptive best result; the developed learned writer specialization did **not** show a positive benefit.

### Per-seed keyed-MLP outcomes

| Writer | Seed 43 | Seed 59 | Seed 71 |
|---|---:|---:|---:|
| Native trained | 53.75% | 55.00% | 58.13% |
| CED trained | 56.25% | 53.75% | 56.88% |
| Native untrained | 57.50% | 55.00% | 57.50% |

Paired differences in keyed-MLP held-out accuracy across seeds:
- Native trained − untrained: **−1.04 percentage points**, exploratory seed-level 95% Student-t interval **[−6.92,+4.84] pp**.
- CED trained − untrained: **−1.04 pp**; observed per-seed negative differences, but only **three independent seeds** and adaptive development mean there is no robust general statistical conclusion.
- Native trained − CED trained: **0.00 pp** mean, with substantial seed disagreement.

The probe's own offline training labels and extra model capacity are *different from* DEV-010 end-to-end training. The probe score must not be interpreted as improved native TAC-OSM decision performance.

### Essential controls and reference

| Control | Mean outcome |
|---|---|
| Zero-memory on the **same probe** and public present input | **50.00% for all 27 probes** |
| Swapped paired memory, keyed MLP/native trained | **44.38%** |
| Swapped paired memory, keyed MLP/CED trained | **44.38%** |
| Swapped paired memory, keyed MLP/untrained writer | **43.33%** |
| Full-history Euclidean nearest-address control | **84.79%** |

The paired-state swap reduces keyed-MLP accuracy below chance, consistent with the model exploiting a *small history-dependent signal*. The zero-memory control at exactly 50% confirms the balanced-pair diagnostic. The Euclidean baseline scanned all H+D addresses per episode; total **86,784 candidate-address distance evaluations over 480 unique episode evaluations**. Unlike the fixed 768-scalar memory, this reference **stores and scans all historical events** and is not a memory-capacity-matched model.

The keyed-MLP native trained writer classified **both** opposite-past pair members correctly in 30 of 240 tested pairs (12.5%); CED-trained 30/240 (12.5%); untrained 33/240 (13.75%). This is limited pair-wise discrimination, not general robust retention.

### Stratified keyed-MLP results (means across writer seeds)

| Stratum | Native trained | CED trained | Native untrained |
|---|---:|---:|---:|
| Retention | 53.33% | 53.75% | 53.75% |
| Fresh acquisition | 57.92% | 57.50% | 59.58% |
| History H=32 | 57.81% | 56.77% | 58.33% |
| H=128 | 53.65% | 55.21% | 56.25% |
| H=512 | 55.21% | 54.17% | 54.17% |
| D=4 intervening writes | 59.38% | 60.42% | 61.46% |
| D=16 intervening writes | 52.08% | 51.56% | 53.12% |
| D=32 intervening writes | 55.21% | 54.17% | 54.17% |

Accuracy is mostly modest and unstable across interference conditions, remains far below the full-history control, and includes role/case-specific variation. The H=512 probe does not show a stable capability gain or monotonic retention guarantee. The full-state MLP's failure is **not proof of absent information**: it has a larger input and only 128 training pairs/240 steps, with no independent capacity sweep.

## What is established, and what isn't

**Supported by this small development experiment:** a simple supervised query-keyed nonlinear readout can use weak information in the 768-scalar state (zero-state 50%, state swap reverses the small advantage). **Not supported:** that DEV-010's learned writer retained *more* task information than its untrained initial state; that a CED-trained memory is superior; that retrieval is robust at H=512 or D=32; that TAC-OSM achieves efficient real future-state prediction; or that a more powerful probe could not decode more.

The almost equal keyed-MLP performance from the untrained writer is especially informative: **the small recoverable signal is not evidence of learned persistence specialization.**

## Prospective next experiment: DEV-012 writer intervention

Isolate the memory transition itself, not another decoder. Freeze a *common probe architecture and training protocol* and compare matched 768-scalar writers: DEV-010 trained/initial writer; reference gated delta; an interference-aware verified/protected-write variant; and ideally a capacity-/operation-matched contemporary delta-recurrent control. Measure native retrieval, frozen-probe extractability, matched-pair accuracy, D/H stress and write/read FLOPs. Include information-theoretic collision controls and failure to retrieve written bits. Avoid using this already-inspected evaluation stream as a confirmatory holdout; reserve a new future unseen domain.

**Decision: negative/limited** regarding progress toward robust TAC-OSM persistence. Preserve original DEV-003–010 results and leave stacked research PRs unmerged.
