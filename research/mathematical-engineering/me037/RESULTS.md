# ME037 — horizon-controlled context-integrated forecasting

A complete model build with native fast/slow memory, a learned full-window context branch and a bounded monotone horizon gate. All 192 adaptation fits and 8 new AR parent fits completed. Each adaptation received 500 joint updates after its shared 1,000-update AR parent training; parent selection/pretraining costs are recorded.

## Main results

Lower MSE is better. Two-seed means, full target sequences and every eligible test origin (stride 1). The five older datasets are development-exposed. The new household final 20% is a separately reserved temporal test; models train on that same house's earlier observations. This is not zero-shot transfer.

| Model | Legacy development MSE | Prospective house MSE | House energy-only MSE |
|---|---:|---:|---:|
| Native memory + learned context | 0.339944 | 0.862641 | 0.547785 |
| Matched context-only model | 0.341093 | 0.867664 | 0.552472 |
| Matched GRU + context | 0.341475 | 0.865298 | 0.550888 |
| AR-only adaptation | 0.347832 | 0.870339 | 0.589338 |

Legacy native gain versus AR adaptation: **2.27%**; registered gate **PASS**. New-house native gain: **0.88%**; separate prospective engineering gate **FAIL**.

At 720 steps, native versus AR adaptation changes MSE by **+0.13%** on ETTh1, **-0.28%** on ETTh2, and **-1.16%** on the new house. Positive means worse. The envelope is an amplitude bound, not a guaranteed loss improvement.

Historical ME036 native macro MSE was 0.346956; ME037 changes it by 2.02% relative improvement. This is an unmatched system-development comparison: model capacity, AR pretraining and optimization exposure differ. Previously scored benchmark outcomes are not fresh evidence.

## Dataset/horizon results

| Dataset | H | Native + context | Matched context | Matched GRU | AR adaptation |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 96 | 0.387446 | 0.386115 | 0.385874 | 0.390657 |
| ETTh1 | 192 | 0.446038 | 0.435690 | 0.442924 | 0.440746 |
| ETTh1 | 336 | 0.480828 | 0.481836 | 0.479107 | 0.482712 |
| ETTh1 | 720 | 0.479610 | 0.484578 | 0.488861 | 0.478972 |
| ETTh2 | 96 | 0.291047 | 0.291505 | 0.291418 | 0.290702 |
| ETTh2 | 192 | 0.368285 | 0.365731 | 0.377441 | 0.378525 |
| ETTh2 | 336 | 0.416191 | 0.419591 | 0.419043 | 0.418686 |
| ETTh2 | 720 | 0.417753 | 0.421598 | 0.419775 | 0.418911 |
| ETTm1 | 96 | 0.344183 | 0.345080 | 0.343411 | 0.357605 |
| ETTm1 | 192 | 0.379219 | 0.386722 | 0.384118 | 0.394898 |
| ETTm1 | 336 | 0.407821 | 0.413020 | 0.412927 | 0.428353 |
| ETTm1 | 720 | 0.473940 | 0.479384 | 0.475053 | 0.488709 |
| weather | 96 | 0.191949 | 0.193154 | 0.192732 | 0.200482 |
| weather | 192 | 0.230122 | 0.232366 | 0.231315 | 0.243759 |
| weather | 336 | 0.282177 | 0.282741 | 0.282315 | 0.295771 |
| weather | 720 | 0.356966 | 0.357287 | 0.356697 | 0.365857 |
| electricity | 96 | 0.192705 | 0.193089 | 0.193491 | 0.202988 |
| electricity | 192 | 0.195009 | 0.194878 | 0.195045 | 0.202625 |
| electricity | 336 | 0.208649 | 0.208600 | 0.209106 | 0.216953 |
| electricity | 720 | 0.248948 | 0.248906 | 0.248851 | 0.258727 |
| appliances | 96 | 0.484092 | 0.487704 | 0.482768 | 0.485796 |
| appliances | 192 | 0.658620 | 0.669159 | 0.667514 | 0.651905 |
| appliances | 336 | 0.885733 | 0.891676 | 0.886985 | 0.904799 |
| appliances | 720 | 1.422121 | 1.422117 | 1.423924 | 1.438858 |

## Prospective energy forecasts

The 20-channel house score includes 18 temperature/humidity channels. Energy results are therefore shown separately. MAE in Wh below is obtained by rescaling each standardized energy residual with its training-only channel std; it is not measured energy savings or a repair/control outcome.

| H | Model | Energy-only standardized MSE | Appliances MAE (Wh) | Lights MAE (Wh) |
|---:|---|---:|---:|---:|
| 96 | Native memory + learned context | 0.554699 | 44.4656 | 3.0560 |
| 96 | Matched context-only model | 0.560494 | 44.0913 | 2.9711 |
| 96 | Matched GRU + context | 0.555187 | 43.6582 | 3.0095 |
| 96 | AR-only adaptation | 0.587877 | 46.6846 | 2.9535 |
| 192 | Native memory + learned context | 0.544971 | 43.8225 | 3.2477 |
| 192 | Matched context-only model | 0.548145 | 42.2017 | 3.0112 |
| 192 | Matched GRU + context | 0.548820 | 42.0212 | 3.0001 |
| 192 | AR-only adaptation | 0.596043 | 45.9096 | 2.9282 |
| 336 | Native memory + learned context | 0.546206 | 44.8567 | 3.3933 |
| 336 | Matched context-only model | 0.554534 | 43.0870 | 3.0825 |
| 336 | Matched GRU + context | 0.552769 | 43.1976 | 3.1128 |
| 336 | AR-only adaptation | 0.591279 | 45.7435 | 2.9288 |
| 720 | Native memory + learned context | 0.545265 | 45.0606 | 3.0452 |
| 720 | Matched context-only model | 0.546714 | 44.4829 | 2.9617 |
| 720 | Matched GRU + context | 0.546777 | 44.6062 | 3.0120 |
| 720 | AR-only adaptation | 0.582154 | 46.9909 | 2.7679 |

## Reproducibility and limits

864 parent/adaptation checkpoints replayed 37,062,144 saved validation values bit exactly. Initial AR forecasts also replayed exactly across 8,236,032 values. These are saved validation arrays, not the entire final test matrix.

48 native models have standardized and raw physical-input standalone exports; training-only anchor buffers are removed without changing learned weights. Export replay covers 4,118,016 values. Physical wrapper replay is against the same eager raw-input pipeline; float32 physical roundtrip is not declared equal to a separately standardized call.

The old 1,680 ME036 checkpoint hashes remain unchanged. Its earlier full replay receipt remains preserved; this turn's old-checkpoint immutability check hashes those files rather than rerunning all their forecasts. The historical ME035B precision exception is unchanged.

All matched neural encoders/readouts differ by less than 1% in trainable parameters and share parents, examples, samplers and selection opportunities. Actual CPU spend differs: **no architectural-superiority claim**. AR is an unmatched practical reference. Cached parent costs are attributed per pipeline but were not physically rerun for every arm; summary.json separates new and deduplicated historical spend. Process RSS is a high-water mark, not isolated activation memory. No full FLOP/energy parity or published-model parity.

Native state is 64 float32 coordinates per channel, not one 64-vector for every channel combined. Serving still replays the whole 96-step normalized window. No arbitrary-lifetime streaming or microsecond controller claim.

The new site is one Belgian house. Its temporal test and two fitted seeds do not establish multi-site population reliability. Energy meters and direct ZigBee measurements are retained; publisher-added random variables and merged airport weather fields are excluded. Previous ETT exposure, shared ETTh1/ETTm1 equipment and Weather anomalies remain explicit.

PR300 was read and remains draft/unmerged; its task-scoped foundation-model comparison is not a general domain prohibition or evidence of TAC action/VoI strengths. No actions, inspection costs, counterfactual repairs, physical control policy or commercial qualification are demonstrated by this study.
