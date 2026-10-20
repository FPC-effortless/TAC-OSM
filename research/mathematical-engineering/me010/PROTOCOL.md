# ME-010 — Event-conditioned native world-state forecasting (pre-registered)

**Date:** 2026-10-20. **Research class:** authentic recorded observations, frozen native TAC world-state recurrence and newly trainable forecast readout. No SSM migration or new CASM planner. No synthetic benchmark success claims.

## Motivation: authenticated failure cases, not invention for its own sake
ME007F March16–23: trained GRU41-1000 beat TAC+direct (MAE 0.331029 vs 0.341877). ME008F March23–30: unchanged native TAC1000 beat GRU41-1000 (0.70078 vs 0.755996), yet **both worse than persistence (0.59548)**. ME009F March30–April6: native TAC1000 beat matched GRU (0.422708 vs 0.431175), and the learned 161-parameter reliability gate reduced quiet false alarms but worsened MAE (0.428613). Thus the real problem is **reliable prediction of changes vs quiet dynamics during distribution shift**, not a missing recurrent-memory equation.

## Mathematically correct two-regime formulation
For each real sensor and future observation horizon h, define the observed supervised event

    E[t,h,f] = 1{abs(actual_future_delta[t,h,f]) > theta_train[h,f]}

where theta_train is computed using only the original early TRAIN slice (first 20 days). This is a *meaningful-change threshold*, not an assertion that quiet states have delta=0. The identity:

    E[delta | history] =
      P(E=1|history) * E[delta|E=1,history] +
      P(E=0|history) * E[delta|E=0,history]

motivates a learnable decomposition; it is a conditional-mean/MSE identity, **not proof of MAE-optimality**. The native predicted delta remains frozen. A causal feature head ingests native delta, native absolute delta, observed historical variability, observed signed trend and absolute trend, current availability mask, observation fraction and horizon. No future label is an inference input.

Event-conditioned model:

    p = sigmoid(head_0(features))
    mean_change = 2 * native_delta + head_1(features)
    mean_quiet = head_2(features)
    predicted_delta = p * mean_change + (1-p) * mean_quiet.

All three output heads initialize at zero so p=0.5, mean_change=2*native_delta, mean_quiet=0; **initially predicted_delta equals the verified frozen native forecast exactly**.

Direct control uses the *same 8->24->3 network and active output channels*:

    predicted_delta = native_delta + (head_0+head_1+head_2)/3.

Thus model/parameter budget is matched for the two readout alternatives, not the full GRU or TAC training budgets. Both use identical real inputs, seeds and optimizer updates. The event model also has extra supervision (binary cross-entropy on real eventual changes and event/quiet conditional Huber), which is the scientific intervention being tested.

## Data and exact original experiment
Original UCI MetroPT-3 publisher archive fixed SHA256 `aab991a970e58210de853bb4d9412fdc5c79792dae3d8e1721a`. Both pretrained native TAC1000 and GRU41 controls come from registered ME008/ME007 original TRAIN first 20 days. **Only** already explored chronological days20–25 can be used for new supervised readout fitting (explicit historical calibration data, NOT new holdout). Same 12 real physical channels, 32 historical observation steps, missingness mask and future readout targets at steps 12 and 60, original normalization and TRAIN-only change thresholds.

Exact three seeds for new readout: 101001/101002/101003 paired with frozen original native seeds 240401/240402/240403. Fit 6 original readouts (three direct, three event-conditioned) for 260 AdamW steps each, batch128, LR0.001, weight decay0.0001, clip grad1. Smooth L1 beta0.2 on all authentic future deltas, plus 0.20 conditional changed-target loss; event arm additionally gets 0.15 binary log loss and 0.10 event-conditional + 0.10 quiet-conditional Smooth L1. Those objectives and weights are frozen before seeing new source. All original native weights have requires_grad=false; archive complete optimizer/provenance, all six readout checkpoint hashes and any failures.

## Reserved first-exposure future period and decision
**April 6, 2020 inclusive to April 13, 2020 exclusive** remains unread by this original fit and every earlier registered study. Score exactly once with ZERO refits. Compare original native TAC1000 (3), GRU41-1000 (3), ME009 gated forecasts (3), ME010 event mixture (3), ME010 direct active-channel control (3), and persistence. All source/checkpoint hashes checked BEFORE opening later data. Primary macro normalized MAE (12 physical features x 2 horizons), secondary significant-change magnitude/direction, quiet false alarms, event probability Brier and cross-entropy, precision/recall and 10-bin ECE; compare event probability with a constant event-rate forecast estimated from days20–25 only. Avoid comparing hindsight-change direction with deployable event detection.

**Gate:** Event arm must improve MAE over both direct readout and original native in all three fixed paired seeds, and event Brier must beat the historical event-rate-only constant baseline, to warrant a *new* independent source. Failure: record exact negative result, do not tune on exposed future period. Even success is one compressor and one week, not learned general entity identity, causal action effects, autonomous System 2, safety guarantee, commercial fitness, or architecture promotion. PNDS Lean admission and neural CASM remain separate evidence-controlled mechanisms.

## Continuity
All earlier unified-plan research ideas (CDL-PHR, BDH, dynamic graphs, episodic/procedural executable memories, role-filler VSA, oscillatory gates, attractors, entity tracking, neural CASM, PNDS evidence) are preserved in the master plan. New components are only promoted when genuine real-world evidence identifies a performance benefit. No automatic merge.
