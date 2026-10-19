# SEQ-TRANSFER-005 — true long-history entity-local sequence modelling

**Executed:** 2026-10-19. **Status:** all 60 preregistered-in-script neural fits completed locally; independent streaming-state audit passed. This is research-only, no architecture promotion, no merge, no checkpoint overwrite.

## Primary scientific answer

**The DEV-061.5 recurrent architecture can preserve a simple early entity bit across 512 complete events, but cannot reliably execute late corrections, repeated toggles, or two-entity parity under the fixed short-history training recipe. Pretrained DEV-061.5 does not outperform training the same architecture from scratch overall.** A smaller, simple *entity-routed GRU* achieves the same perfect stale-state retention without TAC-OSM pretraining or CASM.

This experiment corrects SEQ-TRANSFER-004's central limitation: **long inputs are not truncated to the latest 64 events for the recurrent agents.** An independent test explicitly executes the native trained local GRU transition **one event at a time** and verifies that its final memory matches the compact batched evaluation (maximum absolute difference 2.22e-16 for lengths up to 512). The frozen original checkpoint is SHA-256 verified; the encoder is then fine-tuned on the sequence objectives. **This is a newly constructed streaming readout adapter using learned recurrent/relational weights, not the unchanged original JSON maintenance runtime.** The original 64-event cap is bypassed deliberately. The native public-ID grouping remains a *programmed routing prior*; it is not learned entity discovery.

## 512-event primary results

Held-out accuracy on **four entities**, averaged over two independently initialized supervised training runs per arm; same held-out examples used for both seeds. 200 unique held-out examples per task/length, each seed evaluated on that set. Chance ≈50%. Each task has a separate trained head and encoder fine-tuning; this is not a universal multi-task model.

| Task | Pretrained TAC | Scratch TAC | Routed GRU | Global GRU | Window-64 Transformer | Simple gated SSM |
|---|---:|---:|---:|---:|---:|---:|
| Stale SET (early entity state) | **97.50%** | **100.00%** | **100.00%** | 51.00% | 52.75% | 48.25% |
| Late overwrite (updated state) | 58.25% | 54.75% | 54.25% | 54.50% | 52.00% | 51.50% |
| SET followed by 1–4 TOGGLEs | 49.00% | 50.00% | 51.75% | 48.75% | 50.00% | 51.50% |
| Two-entity XOR | 56.75% | 59.50% | 56.50% | 60.00% | 50.50% | 57.25% |
| IID unpredictable label (negative control) | 52.00% | 53.00% | 50.25% | 54.00% | 47.25% | 53.00% |
| **Mean across five tasks** | **62.70%** | **63.45%** | **62.55%** | **53.65%** | **50.50%** | **52.30%** |

The averages are descriptive, not uncertainty-adjusted significance claims. A 5-gram baseline from SEQ-004 is **not** appropriate for 512-event stale-state identification from an event >128 steps earlier; it would not have the required information. A separate **exact deterministic state-machine reference** reconstructs all labels in four deterministic tasks at 100% and detects any generator/indexing mistakes. IID labels are independent and therefore deliberately excluded from exact-reconstruction assertions.

## Generalization to different lengths and entity counts

| Task / model | L64 / E4 | L128 / E4 | L256 / E4 | L512 / E4 | L512 / E6 |
|---|---:|---:|---:|---:|---:|
| Stale SET: pretrained TAC | 96.75% | 96.25% | 95.25% | 97.50% | 97.25% |
| Stale SET: scratch TAC | 100% | 100% | 100% | 100% | 100% |
| Stale SET: routed GRU | 100% | 100% | 100% | 100% | 100% |
| Late overwrite: pretrained TAC | 85.25% | 76.00% | 56.00% | 58.25% | 54.25% |
| Late overwrite: scratch TAC | 64.50% | 58.75% | 48.25% | 54.75% | 53.50% |
| Late overwrite: routed GRU | 71.50% | 60.00% | 48.25% | 54.25% | 53.50% |
| Toggle: pretrained TAC | 50.50% | 49.50% | 49.25% | 49.00% | 50.25% |
| Pair XOR: pretrained TAC | 50.50% | 53.25% | 54.25% | 56.75% | 50.25% |

**Information age:** all L512/E4 target labels in deterministic tasks depend on one or more informative writes from at least 128 events earlier. Median distance of *last target-changing event*: stale SET 509, late overwrite 398, toggle 324, pair XOR 380 events. No late future label is available in the decoder input. Target-only NOOP records after changes do not refresh the target's bit in the input; random non-target events continue after its informative update. Entity queries are randomly permuted per example.

**Critical nuance:** toggle accuracy is near chance *even at L64*, so failure cannot be isolated to long-term forgetting; the fixed 100-update recipe does not learn parity/toggle dynamics even in-domain. Late overwrite learns partially at L64 but deteriorates at L256 and L512. Hence keep *rule acquisition* and *state retention* as separate engineering problems.

## Protocol

- **Checkpoint:** original DEV-061.5 `split_robust` exported model, SHA256 `7acc30f5e492b0be599a17ae770a10fd8e04108e179d4f781b7d4b2c636f88e7` (trained checkpoint unchanged on disk).
- **Adapter:** uses original local entity-specific GRU, relational/content refinement, CASM transforms, and a fresh 128→64→2 supervised binary head. It bypasses the original 64-event raw-cache limit through a streaming recurrent transition. Fully randomized TAC uses the **same** active architecture; a routed GRU independently learns an entity-local state without CASM; global GRU shares state across all entities. The controlled 64-window Transformer is **window-limited by design** (not a full-context Transformer or a FlashAttention benchmark). Gated SSM is a simple exponential-gated recurrent control, **not Mamba/S4**.
- **Input:** observed stream of SET, TOGGLE, NOOP actions with public per-episode identity routing and bit on SET, no direct target labels. Query specifies one entity or a pair of entities **after** the observation stream. Inputs include no exact hidden-state dictionary or symbolic transition simulator. All recurrent agents process all events, no recorded raw-history cache survives inference other than ephemeral tensors used for efficient training and evaluation.
- **Training:** 1200 independent sequences per task (600 length 32 and 600 length 64), 200 validation length 64, 200 new held-out examples at each test combination L64/E4, L128/E4, L256/E4, L512/E4, L256/E6 and L512/E6. Data-generated seed and split keys are in `run_seq005.py`. Exactly **100 AdamW updates**, batch 48, learning rate 0.003, gradient clipping at norm 1; fresh model/head training seeds **42 and 43**. All 60 fits complete: 5 tasks × 6 model arms × 2 seeds. No early stopping, tuning on test, teacher forcing at inference or favorable-seed selection.
- **Compute/model size:** native DEV061.5 active parameters ≈29,287 (total includes unused maintenance heads). Routed GRU active parameters 10,954; global GRU 28,402; window Transformer 39,314; gated SSM 6,594. Update counts are matched, but active parameters, operations and architecture priors are **not** equal; do not infer efficiency or universal architectural dominance.
- **Negative labels:** separate IID bits independent of the entire input history. Performance fluctuates about chance on the modest held-out sample. Two training seeds share held-out examples, so do not treat model-seed scores as independent 400-sample confidence intervals.

## Audits and failure boundaries

- **Exact physical/state oracle:** checks the final bit for each queried entity reconstructed from *all* recorded SET and TOGGLE events, not only from the target's last raw bit. Verified 100% on all six held-out length/entity settings for four deterministic tasks.
- **Long-gap guard:** median last-informative-event lag at L512/E4 exceeds 300 for all four deterministic tasks. All such labels have at least 128 intervening events (see `consolidate.py`).
- **Packed vs streaming state:** `audit_streaming.py` computes the original `IdentityAgent.local` transition one event at a time and compares hidden states with `grouped_state()` batched compact packed-GRU evaluation on four test tasks. All pass; observed max difference ≤2.22e-16.
- **No deployment equivalence claim:** the adapter, task schema, head and unbounded sequence input **differ** from the original DEV061.5 exported runtime. The experiment assesses weights and reusable architecture, not unchanged production behavior.
- **No unique CASM attribution:** routed GRU without CASM matches or exceeds TAC on stale-state retrieval. No fully trained native no-CASM arm was included here; its absence prevents causal attribution to learned CASM.
- **No strongest-baselines claim:** the 64-window Transformer and simple gated SSM are deliberately restricted controls; larger full-context Transformer, modern selective SSM and trained explicit structured-memory controls remain undone. A programmed event-state machine trivially solves these exact synthetic tasks and is the most relevant upper benchmark.

## Reproduce locally

```bash
python -m pip install torch numpy
cd research/seq-transfer-005
python bootstrap.py  # only needed for GitHub source checkout; restores from research/dev0615/learned-agent.tar.gz
python run_seq005.py --tasks stale_set late_overwrite toggle pair_xor iid --models tac_pretrained tac_scratch routed_gru global_gru window_transformer gated_ssm --seeds 42 43 --steps 100 --batch 48 --train 1200 --val 200 --test 200
python audit_streaming.py
python consolidate.py
```

The completed fits were executed in several otherwise-identical per-task invocations because long batched runs hit execution time limits. `consolidate.py` rebuilds complete `results.json`, `results.csv` and `results_summary.csv` from the preserved per-task JSON artifacts. Runtime tested under CPU PyTorch 2.10.0, NumPy 2.3.5, deterministic kernels with one PyTorch thread. The original trained weights remain immutable.

## Next experiment: distinguish learning dynamics from retaining facts

1. Train on many more updates and curricula of **SET, TOGGLE, NOOP** and compare with a *trainable keyed transition controller* and a compact explicit-state recurrent baseline.
2. Use a new, fully hidden, held-out test distribution with query-target events confined to early history, including adversarial multiple overwrites and 2, 4, 6, 8+ entities. Train/evaluate across at least 5 checkpoint and optimizer seeds.
3. Pre-register separate gates for information retention, update fidelity, composition, costs and compute, and do not imply state-of-the-art sequence modeling from simple stagnant-bit memory.

**Conclusion:** DEV061.5 has an effective, reusable entity-local recurrence **for stable facts**, but the current objective and training regimen do not establish reliable evolving-world-state sequence modelling. The strong stale-memory result is predominantly a structural factorization effect shared by a smaller GRU. Unchanged checkpoint weights are not sufficient to distinguish TAC from structured recurrent controls.