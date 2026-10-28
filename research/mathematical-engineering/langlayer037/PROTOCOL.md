# LANGLAYER037 — genuine-language feedforward-only depth control

**Pre-training registration** (2026-10-29). Candidate trained on an owner-authorized
500K language-model scale; exploratory internal FIT/previously exposed DEV only.

## Hypothesis and control

We know from LANGLAYER036 that disabling the new recurrent read raises
frozen DEV NLL by +0.016889 for the depth-2 block, whereas graph specificity
and history produce ~zero effect. That does not tell us whether an
equally-sized *nonrecurrent* layer could learn the same incremental prediction.

This study retains the complete original CONTEXT026 native, aligned, wave
and concept source-processing engine, including original write semantics.
Replace the upper-layer recurrence and concept graph with STRICTLY TOKENWISE
nonlinear residual blocks, each using a LayerNorm plus independent
up/gate/down linear functions and a learned gain, all at the same token only.
No new temporal, recurrent, associative, attention, cache or source retrieval
capacity is added by these upper blocks.

- Two stages: one FFN residual block, hidden width **48**, final FFN width
  450; 499,661 learned active/deployed parameters (recurrent counterpart
  499,725 at hidden 48).
- Three stages: two independent FFN residual blocks, hidden width **24**
  each, final FFN width 449; 499,684 learned parameters (recurrent counterpart
  499,934 at hidden 24).
- The original CONTEXT026 baseline 500,122 learned parameters is unchanged
  and its prior five seed checkpoints/scores are not retrained.

All alternatives are within the 1% active/deployed parameter parity bound;
the upper-layer hidden widths and layer counts match. Final FFN widths differ
from recurrent controls, because the models must pay for recurrent and graph
weights out of the same total capacity. This isolates a **capacity tradeoff**,
not an exact fully identical parameter allocation.

## FIT and exposed DEV source plan

- Genuine PSF CPython 3.13.0 original source, 13 frozen FIT docs plus one
  authentic UTF-8 FIT supplement; four **already exposed** DEV docs.
- Same SHA256 source manifest, source-owner-offset ordering and
  deterministic 1701–1705 seeds as LANGLAYER035.
- Ten new real fits: 2 arms x 5 seeds x **320 AdamW updates**, B8,
  context 128 and exactly 327,680 publisher next-byte targets per model.
- AdamW lr 0.001, gradient clip=1, no tuning, two CPU threads, no GPU;
  no synthetic training, protected TEST, external spend or independent
  qualification.
- Every original checkpoint has full model, optimizer, Torch/NumPy/Python
  RNG state, SHA-256 checkpoint hash and original sample-order hash; cold
  reload must yield the same 4096 authentic previously exposed DEV NLL,
  top-1/top-5 and Brier.
- Original five-seed scores, P3 practical baseline and hybrid controls
  are frozen as **historical same-protocol references**, never re-fitted.

## Interpreting evidence

Evaluate per seed matched source, exposure, step count and param counts.
Compute actual CPU seconds and RAM separately; matching FLOPs is **NOT
VERIFIED** and no claim of matched-resource architecture superiority
is permitted. Each attempted fit, including failed runs and negatives,
is retained. Do not choose a best seed or best arm using exposed DEV.
Numbers assess short-window byte-prediction under fixed 320-update
exposure only. Long-context memory/persistence, new real-world
generalization, semantic reasoning and full PNDS composition are OPEN.

**Thresholds for discussion:** compute all paired NLL differences
(FFN depth2 minus recurrent depth2; FFN depth3 minus recurrent depth3),
five independent seed signs, sample means/SD, top-1, Brier,
perplexity and CPU. A single-metric small DEV gap is an exploratory
signal; meaningful gains must survive fresh data, broader corpora,
longer training and matched-compute conditions.

No merge/promotion or modification of original LANG035/036 files.
