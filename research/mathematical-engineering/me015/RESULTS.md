# ME-015 — Original real trained mathematical bottleneck diagnosis

**Original run:** successful [GitHub Actions 39398038257](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39398038257), original source code SHA `95d09ce24a287b74f5a2cff223a4d03ece4618ac`, original evidence artifact `12253856564` (`me015-authentic-native-bottleneck-diagnosis`). Original `results.json` SHA256 `fd1c077379d7407eef8c172e9d766a814e51e9cdef5d4edfe9ad53d6e6d4625b`. All 15 original PyTorch checkpoint SHA256 digests independently verified. Follow-on `me015` fitting GitHub workflow was locked to first source SHA before any new-user exposure.

Authentic KuaiRec real source publisher ZIP MD5 `261550d472c48eff4990fb13c0e5bcf7`, observed SHA256 `c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c`. Genuine training 68825 causal windows from users `uid%16==0` before Aug1; **previously inspected** development 106245 windows Aug1–Aug14. No future Aug14–Sep06 user%16==0 reopened. Five fixed arms × three original paired seeds, EXACTLY 135050 active trained parameters each, identical source, batches, 300 AdamW steps and BCE objective. Measured CPU training times are different, so same step count is NOT an equal wall/FLOP budget.

## Historical real developer-diagnostic results (not new independent success)

| Model | Mean Brier across 3 seeds | Mean BCE logloss | Mean CPU train sec |
|---|---:|---:|---:|
| Original native TAC | 0.202491 | 0.591985 | 6.456 |
| TAC with chrono gate bias only | 0.202011 | 0.591035 | 6.488 |
| TAC with causal prior logit only | 0.200502 | 0.586447 | 6.553 |
| **TAC with both changes** | **0.200405** | **0.586179** | 6.510 |
| GRU + identical causal prior logit | 0.201750 | 0.590432 | 6.113 |

At historical dev 3/3 pairs show TAC combined better than GRU with identical prior; original *unanchored* ME014B GRU dev mean was ~0.200715, so the anchored GRU is *not universally stronger* and was retained as separate baseline for subsequent untrained user tests.

## Actual gate measurements on 256 genuine real histories per seed (means)

| Original real checkpoint model | Actual fast gate mean | Actual slow gate mean |
|---|---:|---:|
| Native original | ~0.102 | ~0.0199 |
| Native chrono | ~0.156 | ~0.0463 |
| Native anchor | ~0.080 | ~0.0199 |
| Native chrono + anchor | ~0.129 | ~0.0427 |

These are actual learned, data-conditioned writes rather than sigmoid of bias alone. The native chrono+anchor model also shifted trained output head norms toward slow versus fast state; this is dependence on state, **not evidence of semantically discovered entities or causal action choice**. A sigmoid/tanh convex state recurrence is coordinatewise bounded [-1,1], but neither gradient flow nor forecast reliability is proved by that invariant.

The later authentic independent cross-user ME015F (`uid%16==1`) had engineered TAC mean Brier **0.200786**, anchored GRU **0.202096**; more stringent ME015G (`uid%16==2`) engineered TAC **0.192343**, anchored GRU **0.193348**, original stronger unanchored GRU **0.193621** on 3-seed means, but original GRU won seed14002. Therefore exactly pre-registered all-three seed superiority to **both GRUs** did NOT qualify. Keep failure visible in future engineering.

Next mathematical engineering: improve seed stability/credit assignment in the original nonlinear fast/slow state under authenticated logged histories; compare to both original/anchored GRU and causal user rate with truly equal FLOPs/time budgets. Do not add new expert banks or swap TAC into an SSM. No merge or commercial claims.
