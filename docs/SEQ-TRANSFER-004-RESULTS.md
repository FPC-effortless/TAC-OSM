# SEQ-TRANSFER-004: end-to-end entity-local sequence training against scratch, GRU, diagonal SSM and Transformer

**Executed:** 2026-10-19. **Status:** completed primary matched-update sweep (50 fits), 20 additional completed short-budget fits; one **unfinished** long-budget convergence attempt. No architecture promotion. Original DEV-061.5 checkpoint weights not modified on disk. The pretrained encoder was **fine-tuned**, unlike frozen-readout SEQ-TRANSFER-003.

## Question

Does the original trained DEV-061.5 entity-local encoder **help end-to-end next-bit prediction** relative to the identical randomly initialized encoder and approximately parameter-matched GRU, diagonal linear recurrent state-space and Transformer baselines under a controlled 120-optimizer-update budget?

## Main results

Macro accuracy averages five independently trained synthetic tasks × two optimization seeds; held-out sample sets are shared between seeds, so they are not ten independent test datasets. Each score is an average of ten model/task/seed cells and is not a per-seed population interval.

| Model | Total parameters (including unused native heads for TAC) | Test 32 | Test 96 | Test 256 |
|---|---:|---:|---:|---:|
| tac pretrained | 39,599 | 79.05% | 79.97% | 79.70% |
| tac scratch | 39,599 | 73.67% | 71.88% | 71.22% |
| gru | 39,146 | 77.05% | 75.88% | 74.53% |
| diagonal ssm | 37,314 | 59.52% | 59.00% | 58.48% |
| transformer | 38,306 | 79.67% | 50.50% | 49.52% |
| Supervised 5-gram | — | 90.00% | 88.80% | 89.85% |

**96-event held-out detail**, averaged over optimization seeds:

| Task | TAC pretrained | TAC scratch | GRU | diagonal SSM | Transformer | trained 5-gram |
|---|---:|---:|---:|---:|---:|---:|
| period5 | 100.00% | 85.88% | 99.38% | 71.50% | 52.50% | 100.00% |
| delay3 | 100.00% | 100.00% | 100.00% | 60.38% | 50.88% | 100.00% |
| delay5 | 80.62% | 75.63% | 79.75% | 62.13% | 50.00% | 100.00% |
| xor4 | 68.75% | 48.00% | 51.00% | 50.00% | 50.00% | 100.00% |
| iid | 50.50% | 49.88% | 49.25% | 51.00% | 49.12% | 44.00% |

TAC pretrained vs scratch at L96: **+8.10 percentage points** under the registered local budget. This is a *bounded training-budget advantage* and not a claim of unconditional model superiority. The simple 5-gram is ahead by 8.83 percentage points at L96. On deterministic tasks alone, the 5-gram learns the exact rule. No native text-generation or language benchmark was evaluated.

### Important stability and failure cases

- All three models (TAC pretrained, TAC scratch, GRU) reach 100% lag-3 (both seeds), including test lengths 96 and 256; this does **not** demonstrate a unique pretrained gain.
- On period5, pretrained TAC reached 100% at L32/L96/L256 for both optimizer seeds. Scratch TAC had substantially more variation. The GRU also reached near-perfect scores at L96.
- Lag-5 at 120 updates is not converged: pretrained TAC around 80.6% on 96 events, GRU 79.75%, scratch TAC 75.63%; 5-gram 100%.
- XOR4 is highly unstable: pretrained TAC seed42 at L96 is 49.0%, seed43 is 88.5%. Scratch/GRU stay near chance at this budget; **do not report this as reliable emergent reasoning or CASM attribution**.
- IID random-bit labels remain at chance, including pretrained TAC L96 mean 50.5%; the trained 5-gram test was 44.0% on one 400-example test set, a finite-sample random fluctuation rather than a negative predictor mechanism.
- Transformer uses fixed sinusoidal positions and is trained at L32 only; it fits L32 but fails when exposed to 64 events. This is one small Transformer implementation and training recipe; it is not evidence that Transformers generally fail long-context modeling.
- Diagonal SSM is **a simple diagonal linear-recurrence model**, explicitly **not** Mamba, S4 or a state-of-the-art selective SSM implementation. Its comparison has limited architecture scope.

## Exact protocol

1. **Checkpoints and provenance.** Restore the original first-preregistered DEV-061.5 `split_robust` learned-agent export from GitHub CI workflow artifact / prior sequence ZIP. The SHA256 is `7acc30f5e492b0be599a17ae770a10fd8e04108e179d4f781b7d4b2c636f88e7`; the original runtime demonstration replay passed **4/4 exact action indexes** on the same host before training. A new `TACHead` class loads that state for the pretrained arm and creates an independent 128→2 task head. Same class, randomly initialized from scratch, provides the direct weight-initialization control. Both update their respective encoder and head on next-bit supervision.
2. **Five tasks**: phase-randomized period5, lag3-copy, lag5-copy, XOR of last four bits, and independent IID random next bit. Each has its **own separately trained model** per task and seed; this is not a single universal sequence predictor. Binary labels and bits are complement-paired; XOR4 complements retain their target because parity is even.
3. **Data per task:** exactly 600 training L32, 200 validation L32 (never used to select models), and 400 held-out each for L32, L96, L256. Seed streams are preregistered in `run_seq004.py` and different from SEQ-TRANSFER-003 held-out streams. The same train/validation/test samples and representation window are supplied to all five model arms.
4. **Input contract:** original 38-feature public entity events; typed status event 0, fixed subject-0 key index, target-free bit in event field 36, public event timestamp in field 37. Native 16-dimensional public identity manifest defines two entity slots; only one is used. The native admission gate is deliberately **bypassed** to ensure all models receive the same input window. For L96 and L256, only the **last 64 records** are retained; no model is being asked to remember earlier tokens and these results do not demonstrate 256-event long-term memory. Other baselines get equivalent bit, public clock and entity indicator, not latent TAC state.
5. **Training:** AdamW lr=0.003, weight decay 0.001, global gradient norm clip 1, batch 64, **120 optimizer steps** per arm/task/seed. Two independent training seeds 42/43 (the two optimizer seeds do not represent two pretrained checkpoint seeds). Outputs are softmax binary next-bit logits. Training objective is cross-entropy. No expert labels/teacher, human-chosen action labels, future leakage, or hand-coded forecasting inside the trainable neural models.
6. **Parameter counts:** native TAC 39,599 including heads not used in this sequence probe; GRU 39,146; diagonal SSM 37,314; Transformer 38,306. Equal optimizer updates and nearly equal full parameter budgets do not equalize effective active parameter count, wall time, floating-point operations or hyperparameter tuning; those remain limitations.
7. **Baseline:** supervised fifth-order Markov 5-gram counts based only on 600 L32 train sequences per task, with Laplace smoothing. It is a simple algorithmic statistical baseline and uses the test sequence's last five observed bits, not the future label.

## Short-budget ablation

At **30** optimizer steps per arm/task/seed (fully completed 20 fits), macro L96 pretrained TAC **62.98%**, scratch TAC **59.08%**; difference **+3.90 percentage points**. The difference is smaller than at 120 steps; pretraining is not universally sufficient for fast convergence.

## Interrupted long-budget check (not ranked)

A requested attempt at 400 updates on lag5/XOR4 timed out before completing all matched arms. Exactly **two** pretrained TAC lag5 cells finished; both scored 100% on held-out L96 and L256. The scratch and XOR4 long-budget arms did not finish and **must not be compared against those completed cells**. The partial run log is retained separately. These two completed outcomes show that lag5 training-budget dependence matters, without establishing superiority.

## Methodological and scientific boundary

This study shows a **positive controlled within-architecture pretrained initialization effect** under a finite supervised budget. It does not show a new state-of-the-art sequence model, general intelligence, out-of-distribution language ability, or unique attribution to CASM. Only a single original DEV-061.5 pretrained checkpoint was used, with two supervised training seeds. More pretrained checkpoints, stronger selective SSM and Transformer baselines with tuned length extrapolation, multi-domain corpora, and actually longer stateful memory tasks are needed. The explicit window of 64 records and native entity manifest are material constraints. The 5-gram beats the neural architectures on this exact simple synthetic family.

### Reproduction

The downloadable ZIP includes the exact immutable exported DEV-061.5 checkpoint, original agent source/runtime and demonstration requests, study code, full primary task-specific JSON files, all 50 per-arm aggregate results, 20 short-budget results, sample short-budget prediction pairs, 5-gram calculations and the interrupted-run log. The original GitHub CI artifact and original SHA256 index remain authoritative. **Per-example predictions for the 120-update primary runs were not retained after the per-task process exits; the source and fixed seeds reproduce them.** Primary per-fit accuracies and training losses are saved in the per-task JSON and the aggregate CSV.

```bash
pip install torch numpy
python run_seq004.py --tasks period5 delay3 delay5 xor4 iid --seeds 42 43 --steps 120
python summarize.py
```

Run from the ZIP's study directory, preserving the local `previous/tac_seq_061_062/dev0615-remote-export/split_robust/` path. Host used CPU PyTorch 2.10.0+cpu, NumPy 2.3.5, deterministic algorithms and one torch thread. The published results were obtained via independent task-by-task invocations to keep output logs and checkpoints separate.