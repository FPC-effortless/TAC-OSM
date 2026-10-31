# LANGDEPTH106 exploratory depth/width results

All six fixed-step checkpoints passed source/configuration/weight and replay-evidence checks before scoring.
This is an owner-authorized Tiny Shakespeare experiment; independent research qualification and architectural superiority are not established.

| Arm | Seed | Split | Mode | NLL (nats) | Bits/byte | Perplexity | Accuracy | Targets |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| four | 10601 | dev | reset_windows | 1.541105 | 2.223345 | 4.6697 | 54.190% | 55769 |
| four | 10601 | dev | continuous_carry | 1.518608 | 2.190888 | 4.5659 | 55.036% | 55769 |
| four | 10601 | test | reset_windows | 1.635406 | 2.359392 | 5.1315 | 51.520% | 55769 |
| four | 10601 | test | continuous_carry | 1.615565 | 2.330768 | 5.0307 | 52.149% | 55769 |
| four | 10602 | dev | reset_windows | 1.540544 | 2.222535 | 4.6671 | 54.297% | 55769 |
| four | 10602 | dev | continuous_carry | 1.517142 | 2.188773 | 4.5592 | 55.025% | 55769 |
| four | 10602 | test | reset_windows | 1.655217 | 2.387973 | 5.2342 | 51.407% | 55769 |
| four | 10602 | test | continuous_carry | 1.633859 | 2.357160 | 5.1236 | 52.156% | 55769 |
| four | 10603 | dev | reset_windows | 1.531107 | 2.208921 | 4.6233 | 54.530% | 55769 |
| four | 10603 | dev | continuous_carry | 1.508090 | 2.175713 | 4.5181 | 55.163% | 55769 |
| four | 10603 | test | reset_windows | 1.643822 | 2.371533 | 5.1749 | 51.654% | 55769 |
| four | 10603 | test | continuous_carry | 1.620804 | 2.338326 | 5.0572 | 52.344% | 55769 |
| eight | 10601 | dev | reset_windows | 1.632460 | 2.355142 | 5.1164 | 52.228% | 55769 |
| eight | 10601 | dev | continuous_carry | 1.610486 | 2.323440 | 5.0052 | 52.787% | 55769 |
| eight | 10601 | test | reset_windows | 1.706874 | 2.462498 | 5.5117 | 50.143% | 55769 |
| eight | 10601 | test | continuous_carry | 1.686291 | 2.432804 | 5.3994 | 50.725% | 55769 |
| eight | 10602 | dev | reset_windows | 1.625807 | 2.345543 | 5.0825 | 52.126% | 55769 |
| eight | 10602 | dev | continuous_carry | 1.603272 | 2.313032 | 4.9693 | 52.760% | 55769 |
| eight | 10602 | test | reset_windows | 1.720848 | 2.482659 | 5.5893 | 49.528% | 55769 |
| eight | 10602 | test | continuous_carry | 1.700459 | 2.453244 | 5.4765 | 50.100% | 55769 |
| eight | 10603 | dev | reset_windows | 1.618751 | 2.335364 | 5.0468 | 52.325% | 55769 |
| eight | 10603 | dev | continuous_carry | 1.598526 | 2.306186 | 4.9457 | 52.895% | 55769 |
| eight | 10603 | test | reset_windows | 1.700176 | 2.452836 | 5.4749 | 50.209% | 55769 |
| eight | 10603 | test | continuous_carry | 1.677652 | 2.420341 | 5.3530 | 50.860% | 55769 |

Paired differences are eight minus four; negative NLL differences favor the eight-layer allocation on this split.

| Split / mode | Paired runs | Mean NLL difference | Run range |
| --- | ---: | ---: | --- |
| dev/reset_windows | 3 | 0.088087 | 0.085263 to 0.091355 |
| dev/continuous_carry | 3 | 0.089482 | 0.086130 to 0.091879 |
| test/reset_windows | 3 | 0.064485 | 0.056355 to 0.071467 |
| test/continuous_carry | 3 | 0.064725 | 0.056849 to 0.070726 |

Each target is counted once per score; uncertainty is across three runs on one fixed corpus, not individual bytes.
Widths differ to match weights. Eight layers hold twice as many per-layer cache states. CPU timings, state bytes, training resource receipts and raw generated samples are in RESULT.json and samples/.
GPU speed and complete FLOPs were not measured. Generated text has no alphabet filter; raw .bin files are authoritative and .txt files replace invalid UTF-8 for display.
TEST access: one declared final scoring pass
