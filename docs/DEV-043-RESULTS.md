# DEV-043 — causal-order head transfer and frozen writer calibration

**Audited adaptive development experiment — not confirmatory or a full-world-model claim.**

* Preregistered development contract: [research/dev043/protocol.json](../research/dev043/protocol.json)
* Source checkpoints: DEV-041 training commit `4ba2ea65cd3d4f3961277d80c49d7fb641f556fc` (GitHub run 39003550880); all encoder, recurrent writer, query/read adapter tensors were frozen throughout DEV-043.
* Original three-seed fit and output: [GitHub Actions 39006497054](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39006497054), source `e9d7658aa99685630e8e6b3046d6d588f5553702`. Original end-to-end run **failed its over-strict bitwise numerical audit**; the three seed fit jobs and 1,210 tests passed.
* Separate audit of the unchanged original per-seed outputs: [GitHub Actions 39007536121](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39007536121) **PASSED**. The later audit aligns CPU thread/determinism settings and records all replay deltas, verifies every downstream ranking, classification and exact simulator physical outcome.
* Archive: [complete audited original DEV-043 evidence ZIP](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39007536121/artifacts/12091453567), SHA-256 `b0d075b337b33069bf0a098f8c146a1ee29fb86812c9d9dd03377bf50ad7b7f7`, 1,629,055 bytes.

## Fixed design and no leakage

Three original DEV-041 seed memories (4101, 4102, 4103) were **not retrained**. Their common 44-dimensional frozen pre-head features were used for (i) existing head without calibration, (ii) positive monotone temperature/Platt calibration on newly generated *order-calibration* examples only, and (iii) head-only fits with 0%, 10% or 50% order-dependent training examples at a constant total 180 paired examples and fixed 120 AdamW updates. Method selected by held-out calibration NLL, action threshold chosen by calibration physical cost. Each seed evaluated 144 new order pairs / 288 worlds, including 96 mature pairs / 192 mature worlds; all formerly inspected DEV-042 worlds forbidden and hash-disjoint. The non-neural last-event baseline was fitted from this study's disjoint calibration data.

**Primary endpoint:** signed, threshold-free correct pair ordering in mature target-wait risk. A deterministic classifier restricted to identical last per-entity observations cannot exceed 50% balanced pair hard accuracy or distinguish within-pair ranking. **Secondary:** hard accuracy, service predictions, failures, unnecessary services, calibrated physical cost and retrospective oracle headroom.

## Results: frozen learned-writer MTSK

| Readout | Ranking seeds 4101 / 4102 / 4103 | Mature hard accuracy (all seeds) | Cost/entity (all seeds) | Failures and unnecessary service / seed |
| --- | --- | --- | --- | --- |
| Original | 100% / 100% / 100% | 50% | 0.552083 | 0 failures; 96 waste |
| Calibration only | 100% / 100% / 100% | 50% | 0.552083 | 0 failures; 96 waste |
| Refit, 0% order | 100% / 100% / 100% | 50% | 0.552083 | 0 failures; 96 waste |
| Refit, 10% order | 100% / 100% / **0%** | 50% | 0.552083 | 0 failures; 96 waste |
| Refit, 50% order | 100% / 100% / 100% | 50% | 0.552083 | 0 failures; 96 waste |

The 10% seed4103 readout **reversed** the original direction on every mature pair under tiny score changes despite identical physical actions. Therefore, scoring all seeded models as unambiguously retaining reliably calibrated causal information would be incorrect.

For comparison, the exact **public-history state reconstruction** achieved 100% pair ranking and mature hard accuracy, **0.510417 cost/entity, zero failures, zero unnecessary service**. The freshly calibrated last-event table achieved 50% ranking, 50% hard accuracy, 0.552083 cost/entity and 96 unnecessary services, identically to the trained recurrent policy's observed actions.

## Random frozen MTSK: not a well-tuned reservoir control

| Readout | Mature ranking across seeds | Mature hard accuracy | Cost/entity across seeds |
| --- | --- | --- | --- |
| Original | 100% / 100% / 100% | 50% | 0.814236 / 0.839410 / 0.846354 |
| Calibration only | 100% / 100% / 100% | 50% | 0.809896 / 0.849826 / 0.847222 |
| Refit, 0% | 100% / 100% / 100% | 50% | 0.833333 / 0.860243 / 0.828125 |
| Refit, 10% | 100% / 100% / 100% | 50% | 0.843750 / 0.833333 / 0.828125 |
| Refit, 50% | 100% / 100% / 100% | 50% | 0.835938 / 0.864583 / 0.825521 |

Frozen random cores did not receive a gain/decay or readout budget search; they remain *one poorly tuned candidate reservoir*, not evidence of general random reservoir insufficiency.

## Retrospective pair-threshold headroom (not a deployable controller)

An explicitly hindsight-labeled diagnostic chooses a different action-selection threshold separately for each evaluation **pair**, using its future ground-truth labels. For trained MTSK, the lowest such reported per-pair headroom cost across the tested variants and seeds was **0.536024/entity** (seed4102, refit-10%), **still above the exact-history reference's 0.510417**. The unchanged original trained heads had hindsight costs 0.551649, 0.549479, 0.551649. Therefore, under this particular single-threshold action policy, a different threshold alone cannot fully recover the decision optimum even with evaluation-label hindsight. It does not rule out a richer action/decision policy.

## Audit correction and integrity

The first independent replay used different CPU execution settings and rejected any difference exceeding 1e-7. The audit-only rerun used the original runner's single-thread deterministic PyTorch settings and logged *all 30* seed × architecture × readout probability checks; the maximum deviation was **1.1920928955078125×10^-7**, 21 individual scalar differences above 1e-7, and **none above 1e-5**. The audit then reconstructed the fixed-pair target labels, *every reported model prediction within float32 tolerance*, hard prediction metrics, pair-ranking metrics, and real physical simulator interventions with the **same discrete results and costs**. Full regression suite: **1,210 tests passed**.

This is a reproducibility correction, **not** new writer training, cohort replacement, or retrospectively selected scientific result. Original failed CI runs remain visible.

## Decisions and limitations

1. **Do not promote** calibration-only, the 10% or 50% head, or masked MTSK as a solved order-conditioned world model; the mature hard-accuracy objective is still 50% for every tested trained-head variant and physical cost never beats the last-event table.
2. Perfect rank is not enough. The original trained and random recurrent cores already ranked all pairs; head refitting sometimes only changes the score margin, and in one case reversed the direction.
3. Existing synthetic task gives strong explicit public IDs, and a compact order-aware table still solves it. There is no evidence neural model complexity earns its cost.
4. The user's full requested menu (**isotonic calibration** and **learned schema41 head**) is **not** part of this bounded DEV-043 pilot. Isotonic can create ties on tiny margins; register separately rather than add it after inspecting this result. A gain/decay-scaled frozen reservoir and matched readout/compute tuning were also **not run**, and remain mandatory before broad writer-learning claims.
5. Next controlled studies: DEV-044 gain/decay/equal-budget frozen-reservoir convergence; a separate causal-order-trained action-selection experiment with a head that explicitly optimizes decision costs; then benchmarks with unknown/large sufficient state where trivial tables cannot solve the task. Retain fresh unseen domains and independent multi-seed gates.

**Bottom line:** The central failure is not decisively attributable to missing recurrent information, nor solved by simple calibrators. There is a usable yet often extremely weak ordering signal; converting it to selective, low-cost action remains unsolved.
