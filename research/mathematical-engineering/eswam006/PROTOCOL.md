# ESWAM006 — preregistered factual long-history motion baselines (no model fitting)

**Freeze BEFORE scoring real traffic DEV outcomes. October 23, 2026.** Original authenticated published INTERACTION TC source SHA256 9aa78102006dc8d8a64675cf64eba5264085cfb1c006b1f4d0e856d94e657aec. Only authentic real observation rows, true original agent IDs, original measured timestamps and actual subsequent positions. No artificial distractors, missed-frame interpolation, test-time future velocity, extra synthesized observations, pseudo-action interventions or modeled sensor data.

## Fixed source split and eligibility

Source camera site TC_BGR_Intersection_VA, original publisher recorded-track files 000/001/002 withheld for *future prospective FIT* and file 003 for exposed DEVELOPMENT forecast scores; the member split is by physical recording, not random overlapping windows or actor. All four recordings originate from ONE physical intersection location, so no independent location generalization or untouched complete TEST. The previously executed source-only qualification inspected metadata/counts in all four files, without fitting or scoring future x/y. No source keys across recordings. No data-dependent tuning or reselecting eligibility by observed future accuracy.

For this first actual long-memory task, use every original vehicle trajectory in recorded original file 003 that has 256 complete consecutive historical frames and then a factual available future of 10 or 30 frames, with strictly increasing original recorded timestamp_ms and frame_id increment 1. Source measured present x/y and vx/vy are required and finite. No interpolation, padding, selected synthetic intervention or generated distractors. Start origin after 256 measured history frames. Evaluate every tenth eligible origin (stride10) to limit redundant case multiplicity; still group analysis by track ID rather than pretending frame samples are independent.

## Registered fixed predictions

Let real source present position p_t=(x_t,y_t), logged measured historical velocity v_t=(vx_t,vy_t), and delta_t be actual source *previous* timestamp interval, not future observation. Nominal prediction times are 10*delta_t and 30*delta_t (1s/3s when publisher interval 100ms, but disclose actual interval statistics). The source future true p[t+h] is a scored target, never a predictor feature. Evaluate:
- Persistence P0: predicted p[t+h] = p_t.
- Last measured instantaneous velocity P1: p_t + h*delta_t * v_t.
- Eight-history observed velocity mean P8: p_t + h*delta_t * mean(v[t-7:t]).
- 256-history observed velocity mean P256: p_t + h*delta_t * mean(v[t-255:t]).
One static formula each, no fitting, weights, source-specific threshold tuning or learned checkpoint. Strong last-velocity and persistence controls must remain visible even if long history helps. Consecutive history and future on the same real track, nothing fabricated.

Source target error = Euclidean L2 physical displacement error in meters and mean |x/y| coordinate MAE in meters, separately at future horizons 10 and 30 frames. Report original *track* and *query* denominator and original actor ID/recording scope, each horizon, all four baselines. Confidence uncertainty, if computed, must sample whole actor IDs and avoid claiming fresh independent physical locations. The mere ability to use 256 observed steps does not establish trained neural long-memory capability.

## Interpretation and promotion

If P256 improves on P8 and instantaneous velocity on real original future x/y positions, this supplies evidence that long factual velocity history can sometimes help a simple predictor under fixed formulas; it does not demonstrate that ES-WAM has learned or retained that information. If P256 loses, that negative result sets a strong reference for justifying any more complicated learned memory system. Source tracks have multiple simultaneous actors but this first baseline evaluates one actor at a time and does NOT prove learned cross-actor interactions, entity collision prevention, causal action intelligence, safety or customer value.

Next scientific step, only after reviewing original results: prospective matched 100k-parameter native exchange-memory, GRU and causal Transformer/attention model, identical held-out original recording, source windows, optimizer/example/FLOP budget. Evaluate real multi-agent actual predicted future positions with/without 256-frame history, direct source identity and memory ablation. No claim that traffic observations include robot commands, interventions or verified decision costs.
