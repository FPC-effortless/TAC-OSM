# TAC-LANG-GRAPH-006 — 500K concept cancellation, streaming and bounded-write repair

**Date:** 2026-10-28. **Status:** source-correctness repair in a **separately versioned, untrained** model. The original PR #528's historical 499,290-param model, 503,514-param historical head, 500K real-text checkpoints and **negative real historical retrieval results are unchanged**. Do not compare a new randomly initialized model to a trained legacy model or claim improved predictive quality from numerical operator tests.

## 1. Confirmed exact graph cancellation in the old equation

The six-node graph is symmetric: `A=.5*(softmax(QKᵀ/4)+transpose)`, `L=D-A`, so `1ᵀL=0`. The old solve is

```math
(1.25 I+0.25 L)C=G+0.25 C_{prior}.
```

Summing all six node equations and dividing by 6 yields

```math
mean(C)=0.8 mean(G)+0.2 mean(C_{prior}),
```

**independent of all off-diagonal graph interactions and the learned `concept_q.weight`/`concept_k.weight` (2×256=512 coordinates)**. The legacy `pooled=C.mean(-2)` was fed to the decoder, so cancellation was structural, not merely small gradients. A finite-gradient check allowed zero numerical derivatives and missed the bug.

## 2. Versioned graph-visible v2, with the same 499,290 learned parameters

`tac_wave_ced_500k_v2.py` copies the exact stream model on PR #528 and explicitly changes:

```python
C = solve(1.25*I + 0.25*graph_scale*L, G+0.25*prior)
read_weight = softmax((G*prior).sum(-1)/4 + linspace(-1.5,1.5,6))
pooled = sum_slot(read_weight* C)
```

These fixed slot biases break the uniform symmetry while the content term makes them query-dependent. They introduce **zero learned parameters**; the learned `concept_q/k` remain wired into `L` and now possess a differentiable route into logits. `graph_scale=0` provides a controlled graph-off counterfactual **in the exact same weights**. The final trained checkpoint cannot be substituted into the v2 semantics and called already validated: the source now differs and needs retraining under matched baselines.

The source-specific unit suite must show **logits differ when graph_scale changes**, **nonzero numerical gradients for both concept_q and concept_k**, and **perturbing either learned matrix changes logits**. It explicitly rejects merely finite but zero gradients. The old source is unmodified; v2 graph invariance is not claimed for all degenerate inputs (e.g. indistinguishable concept slots).

Lean module `PNDS/LanguageWaveGraphRepair.lean` establishes the zero-sum cancellation conditional and a two-concept analytical witness: graph solution `(24/35,4/35)` vs no-graph `(4/5,0)`, same arithmetic mean, but read weights `(3/4,1/4)` change output from `3/5` to `19/35`. This is a **precise mathematical counterexample**, not a full six-slot/Torch source correspondence proof.

## 3. History query validity based on global source ordinals

The old `historical_500k.py` incorrectly checked `lag < local_chunk_length` and sliced `[:,lag:]`. That rejects a lag192 query following a 200-token carried prefix when the new chunk has length one and also drops later valid predictions.

The **separately versioned** `historical_500k_v2.py` inherits the v2 base and instead computes

```text
first_valid_local = max(0, lag-start, lag-key_shift-start)
valid_abs_positions = start+first_valid_local, …, start+T-1
target_absolute_position = current_absolute_position-lag
```

It reads only the *pre-write carried source memory*, never the historical target byte, and optionally returns per-lag current and target absolute ordinal metadata and local valid start. A chunk containing no valid queries returns a correctly shaped empty distribution rather than throwing for lack of local length. A negative/zero lag, future-shifted key, wrong carried count, or missing carried state fails closed.

Important: mathematical chronological validity does not authorize cross-user source reads. The original `count`-only state carries **no source owner ID**; owner-scoped externally authorized state storage remains an open production qualification requirement.

## 4. Fix uncontrolled write accumulation without claiming perfect recall

In the old native and aligned source banks, `alpha=sigmoid(retention)` and `write=sigmoid(write_logit)` are independent. As alpha→1, the source write Q can continue adding material faster than it is forgotten, despite each update's state-difference transition being nonexpansive. A bounded tanh value and unit key alone cannot provide a uniform global state bound.

The v2 adapter uses exactly the same learned gates and parameters but changes

```text
write_raw = original_write_sigmoid
write = (1-alpha)*write_raw
0 <= write <= 1-alpha.
```

For a single head with normalized key `||k||≤1`, erase η∈[0,1], retention α∈[0,1], bounded input value `||v||≤V`, the original per-head affine operator is `P=alpha(I-eta k kᵀ)`, `Q=write v kᵀ`. Its spectral norm obeys `||P||₂≤alpha`; `||Q||_F≤write V`. Therefore if `||M||_F≤V` initially, then
```text
||M P+Q||_F <= alpha V+(1-alpha)write_raw V <= V.
```
For source `tanh` value vectors the real-arithmetic bounds are native `V=√8` and aligned `V=4` per head. This is a conditional algebraic design justification with numerical 320-token-chunk tests, **not an IEEE/GPU universal invariant**. The Lean module separately proves the scalar interval version. Complete 16×8 Frobenius matrix proof and finite-precision error accumulation remain open. Coupled write may improve boundedness but **reduces write strength and can harm memory retention**.

Crucially, even bounded state can erase old content: when alpha=0, the next state is independent of the old value. Exponential fading can continue at alpha near one. **This design does not prove recoverability or source recall**, which still failed original real-data gates.

## 5. Qualification gates

1. The v2 source, diagnostic graph-off intervention, exact parameter count 499,290 and history-head 503,514, 200+1-token and 185+21-token streaming queries, gradient norms and 320-token bounded-state tests must pass reproducible cold PyTorch CI.
2. PNDS must pass strict theorem + axiom auditing with no sorry, plus full root Lean/Mathlib build; boundedness assumptions and full graph/source theorem gaps must remain explicit.
3. Train **separate** new v2 and original-model controls on the same genuinely authorized real text, with matched 475K–525K active learned budgets, comparator gap ≤1%, data/sample hashes, optimizer updates, explicit compute cost and replayable checkpoints. Compare graph-v2 versus **the same v2 graph_scale=0 trained ablation**, not just original source-changed checkpoints. Require lower held-out NLL and better specific historical-source outcomes, not merely changes to logits or gradients.
4. Record actual retrieve/erase tradeoff on real source IDs, wrong-owner/ordinal and no-memory tests, calibration, training throughput and streaming batch-1/8 latency. Independently sealed TEST, authorization, cold checkpoint replay and external source storage remain open. Original negative experiment evidence must not be overwritten.

**Decision:** promote neither trained ability nor product claim. This phase corrects graph mathematical wiring and local stream/state contracts; whether either improvement helps genuine language learning is still unknown.
