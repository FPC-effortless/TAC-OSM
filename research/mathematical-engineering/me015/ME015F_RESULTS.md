# ME-015F — Authenticated first NEVER-trained real-user results

**First-exposure source/results:** successful [GitHub Actions 39399405951](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39399405951); original source head commit `c2f24ab6910549731099925681368d951992e918`, original artifact `12254846719` (`me015f-original-first-never-trained-users`). SHA256 of complete original `first_disjoint_user_results.json`: `d254d3b56a32b779f46a87590b2c60288ce766395bd2a4cc1fbe0da062c48783`. Source workflow frozen after success, no rescore claimed as untouched. Original 15 exact-capacity checkpoints from successful [ME015 run 39398038257](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39398038257), artifact `12253856564`, source training original commit `95d09ce24a287b74f5a2cff223a4d03ece4618ac`; **all 15 checkpoint SHA256 verified** before newly reserved users opened.

**Authentic data and disjoint proof:** Real KuaiRec publisher ZIP MD5 `261550d472c48eff4990fb13c0e5bcf7` / SHA256 `c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c`, big_matrix only. FIT and historical DEV from users `uid%16==0`; new never-trained evaluation `uid%16==1`, genuine historical same-user interactions (current reaction excluded from inputs) and future dated exposed videos Aug14 through Sep05 2020 inclusive, first newer evaluation **41,923 valid video-response prediction windows from 439 actual untrained users**. Zero gradient updates/test fitted weights, no post-Sep06 response labels read. Pseudonymous IDs only organize causal history, never enter model.

| Real frozen model (3 seed mean) | Brier, lower better | Bernoulli log-loss |
|---|---:|---:|
| Train-only global full-watch rate | 0.220631 | 0.633425 |
| Train-only item/video rate | 0.219398 | 0.630418 |
| Actual new-user prior32 full-watch rate | 0.202692 | 0.605039 |
| Original native TAC | 0.202300 | 0.591604 |
| Timescale-initialized native TAC only | 0.201912 | 0.590874 |
| User-logit anchored original native TAC only | 0.200929 | 0.587394 |
| **Timescale + anchored native TAC** | **0.200786** | **0.586991** |
| GRU + identical historical logit anchor | 0.202096 | 0.591607 |

**Paired-seed values (engineered TAC vs anchored GRU):**
- 14001 **0.200591** vs 0.202315;
- 14002 **0.201159** vs 0.202061;
- 14003 **0.200608** vs 0.201913.
Native engineered TAC wins in ALL three pre-registered paired seeds and beats the true causal prior32 user-rate baseline. The gain is ~0.65% relative to the anchored exact-parameter GRU and ~0.75% versus original native recurrence. All 15 neural weights contain exactly 135050 active parameters and share authentic FIT, three seeds, batch128, 300 fixed AdamW updates and identical real temporal samples.

**Important limitations:** This is **cross-user** OOD source generalization on the same underlying platform, not a causal action intervention and not unrelated-domain robustness. Training CPU seconds averaged ~6.51 TAC vs ~6.11 anchored GRU in original ME015 fit; exact trainable params/optimizer updates do NOT imply equal FLOP/time budgets. The original ME014B unanchored GRU is another strong registered conventional control that was not scored on this new user cohort; consequently TAC's present superiority is **relative specifically to anchored GRU**, not all GRUs. A separate fully untouched disjoint user cohort `uid%16==2` can compare both frozen original unanchored and anchored GRU controls with the engineered native model, before any new source outcomes are examined. Do not select architecture based on this result alone.

**Mathematical diagnosis:** Chrono initializations increased actual mean slow gate updates from ~0.0199 to ~0.0427. Causal historical prior makes TAC's recurrent state specialize in the residual predictive information the simple user mean omits; its trained slow/fast head-norm ratio grew substantially, but norm does NOT prove retained semantic state. Stronger next tests should assess long-delay retention and strict equal compute, preserving original failed and favorable findings without converting TAC to SSM. No merge/commercial decision qualification.
