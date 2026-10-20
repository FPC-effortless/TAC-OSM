# ME019F — ORIGINAL FIRST-EXPOSURE frozen unseen real users: verified results

**Successful original once-frozen Actions [39446897691](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39446897691)**; original [artifact 12275648767](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39446897691/artifacts/12275648767), zip digest SHA256 `a53c5033544c756a21a24d2924127df901f224bd327f0ab62a8654e63db7db64`, original raw result `results/me019f/unseen_user4_original.json` SHA256 `10e53db93efee995b327643e7f427741c4c2775bfd1cf000f57cbe38c4fbc588`.

**Critical irreversible provenance:** ME019 original 18 pretrained genuine KuaiRec fitted checkpoints from successful original Actions run 39443325909, raw training `results.json` SHA256 `2a10ae3205e866ed792f5ad145e3c3c775e95c39de675f421184de44e56455b6`, ALL 18 pretrained checkpoint SHA256s verified **before opening** the user4 outcome source. Publisher ZIP SHA256 `c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c`. Training user%16=0 FIT originally, no user4 outcomes read in ME019. The original preregistration was committed before user4 first exposure. No optimization, validation tuning, reseeding, calibration or model architecture change on user4 after freeze.

**Authentic real first test:** 43,835 actual video impression/watch outcomes from **442 new users** whose IDs satisfy `uid % 16 == 4`, current timestamps Aug14–Sept06 2020, **128 actual causally prior same-user interactions**, exactly shared most-recent 32 between history-length arms; no synthetic data, history padding, or current-response input. Each of the 18 models has 135,050 active trainables and remains frozen.

## Primary genuine first-exposure results

*Brier lower is better.* Exact original seed values and across-seed arithmetic means.

| Architecture | Real history T | Seed 14001 | Seed 14002 | Seed 14003 | Three-seed mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| native_chrono_anchor | 32 | 0.198005298 | 0.197859169 | 0.196510454 | **0.197458307** |
| native_chrono_anchor | 128 | 0.197946602 | 0.198526420 | 0.197827067 | **0.198100030** |
| gru_original | 32 | 0.197564123 | 0.198215003 | 0.197592350 | **0.197790492** |
| gru_original | 128 | 0.196375287 | 0.196998680 | 0.197053250 | **0.196809072** |
| gru_anchor | 32 | 0.197530638 | 0.198699196 | 0.198856653 | **0.198362163** |
| gru_anchor | 128 | 0.197126680 | 0.198699204 | 0.198856643 | **0.198227509** |

Causal history-based prior32 rate baseline Brier **0.19777338598** and log-loss 0.595376. Mean model log-loss native32 **0.579855**, native128 **0.581524**, GRU-original32 **0.581814**, GRU-original128 **0.579480**, anchored GRU32 **0.582523**, anchored GRU128 **0.582270**.

### Prespecified user-clustered paired Brier contrasts (positive: native128 worse)

Each cluster bootstrap resamples actual **442 users** with 1500 replicates, seed 19019. All values from immutable original real results JSON.

| Contrast | Model seed | Paired mean Brier delta | 95% user-cluster bootstrap interval |
| --- | ---: | ---: | --- |
| native128 − native32 | 14001 | **−0.00005870** | [−0.00053894, +0.00042874] |
| native128 − native32 | 14002 | **+0.00066725** | [+0.00038912, +0.00094680] |
| native128 − native32 | 14003 | **+0.00131661** | [+0.00095841, +0.00170544] |
| native128 − original_GRU128 | 14001 | **+0.00157132** | [+0.00043967, +0.00262001] |
| native128 − original_GRU128 | 14002 | **+0.00152774** | [+0.00075067, +0.00223425] |
| native128 − original_GRU128 | 14003 | **+0.00077382** | [−0.00019913, +0.00171931] |
| native128 − anchored_GRU128 | 14001 | **+0.00081992** | [+0.00026212, +0.00135309] |
| native128 − anchored_GRU128 | 14002 | **−0.00017278** | [−0.00095065, +0.00058155] |
| native128 − anchored_GRU128 | 14003 | **−0.00102958** | [−0.00151049, −0.00058526] |

## Preregistered conclusion and mathematical-engineering decision

- H1 (native128 beats native32 in **every seed**): **FAIL.** Native128 deteriorates 2/3; seed14001 tiny improvement CI includes zero. Across-seed mean degrades +0.000642 Brier. Original ME019 previously inspected DEV native128 worsened 3/3, +0.001212 mean. **Longer original TAC state has failed to generate a robust generalization gain**.
- H2 (native128 beats original strong GRU128 every seed): **FAIL.** GRU-original128 has lower error 3/3, although user-clustered interval includes zero for seed14003. No proof of universal GRU dominance or fully controlled advantage.
- H3 (native128 beats causal user prior32 and anchored GRU128 in each seed): **FAIL.** Mean native128 Brier 0.198100 is above prior32 Brier 0.197773. On direct paired model comparison, native128 beats anchored GRU128 2/3, not all seeds.
- H4 (native128 gains by additional history while GRU does not): **FAIL.** Original GRU128 improves 3/3 vs its own original GRU32.
- The main ME019 real-data **generalization failure replicated sufficiently to prioritize mechanistic diagnosis**, but **the mechanism is NOT proven**: 128-step gradients, Jacobian spectra, state saturation, gate decay, stale-memory pollution, entropy and interference remain unmeasured causal explanations. No architecture conversion or new neural gate can be promoted solely on these Brier scores.

**Research rules remain compulsory:** matched 135050 active trainables and train targets/seed/update budget, but **NOT equal observed training CPU/FLOPs/wall time**; claims comparing architectures are descriptive/engineering only until strict compute-matched replication. Source is logged observational reactions, not action counterfactuals, independently verified outcomes or autonomous commercial control. No new-user reruns to tune hypotheses and no claims of actual causality from memory knockouts.

**Next:** ME020 original early FIT-only frozen 32/128 real-state diagnostic, measuring exactly what failed (128-step age-gradient sensitivity, real gate saturation, retention products, readout reliance) rather than blindly adding attention, Titans/MIRAS or a new decay rule. Preserve successful native 32 checkpoint as research reference; keep negative ME019 and ME019F PRs draft/unmerged, no production promotion.
