# ME-014BF — First frozen real action-conditioned memory comparison

**Scientific decision:** The registered hypothesis that native TAC persistent memory improves observed full-watch prediction over a strictly equal-parameter GRU is **REJECTED** by authentic heldout data. Both memories learn meaningful historical signals, but the simpler structured prior-32 user event rate beats native TAC on mean Brier, and GRU beats TAC on all three paired seeds with less CPU time. No architecture promotion, production merge or action-policy claim.

## Immutable original provenance and data integrity

- **Successful first-exposure test:** [GitHub Actions run 39395321513](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39395321513), original source SHA `5dea9bbbee43e25a9719e28df10225da66fb2e37`, original artifact ID `12251029321` (`me014bf-first-frozen-exact-user-outcomes`). Exact `future_scores.json` SHA256 `12f01b59d34826eae34fdecd20596a131bf9dcfb9d076d3507971af2b7021d02`. Frozen evaluation CI now SHA-locked; further PR edits must not repeat as untouched.
- **Successful original strict equal-size fits:** [run 39394500641](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39394500641), original source SHA `f52aa90dd4061bf84ed915e09a773a4526b30db1`, archive artifact ID `12251666454` (`me014b-exact-real-original`). All six PyTorch trained checkpoint SHA256 digests independently verified from original archive, and verified again *before* future reaction CSV labels parsed. Original model source is now SHA-locked against silent refits.
- Original publisher: authentic KuaiRec.zip from Kuaishou platform observations at Zenodo record 18164998, official original MD5 `261550d472c48eff4990fb13c0e5bcf7`, independently recorded ZIP SHA256 `c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c`, archive **431,964,858 bytes**. Genuine `big_matrix.csv` only; 12,530,806 source rows scanned. Fixed user cohort `user_id % 16 == 0`; 449 users, 576,996 valid source reactions through Aug13. Original source did not parse 205,653 additional future rows until after model weights frozen.
- FIT users and video responses before 2020-08-01: 68,825 eligible causal 32-history training windows; historical engineering diagnostics from Aug01 through Aug13: 106,245 windows. **First future cohort** Aug14–Sep05 inclusive (end Sep06 exclusive) was unopened at fitting. Final test: **41,131 valid future video exposure/response windows from 433 actual users**. The later evaluation parsed 205,653 real post-development cohort interaction rows (only 41,131 selected via deterministic pre-registered example cap). No source rows at or after Sep06 parsed.
- Shared trainable action/video ID embeddings; current offered video and known duration as action; 32 earlier actual same-user impressions and revealed binary completed-watch outcomes; no current target or future reaction enters predictor. User ID groups histories in the audit harness but is **not** a neural input; thus no emergent entity identity discovery claim.

## Strict matched trainable-parameter and training-data controls

Both native two-timescale TAC and GRU with an active nonlinear action-conditioned residual contain EXACTLY **135,050 trainable parameters**, not dummy padding. Same genuine 68,825 train windows and labels, three paired seeds 14001/14002/14003, 300 fixed optimizer updates, identical sampled batch-index seed, batch128, **Bernoulli BCE-with-logits** objective, AdamW LR .001/decay1e-4 and gradclip1. The original near-matched 135050 vs 134982 run remains archived separately as nonstrict engineering evidence; only ME014B is used for this comparison.

Original measured training CPU seconds, 3 seeds native **10.23 / 10.07 / 9.98** vs GRU **7.18 / 7.12 / 7.11**. Mean native training ~**1.41× slower**. Equal active params, data and number of optimizer steps do **not** mean equal FLOPs/wall-time training budgets; a strict equal-compute superiority result remains unavailable. It is nevertheless clear TAC has NOT shown a performance or efficiency advantage over this GRU control.

## First untouched future real reaction outcomes (lower Brier and BCE better)

| Predictor (3-seed mean where relevant) | Brier score | Binary log loss |
|---|---:|---:|
| Original training-only global complete-watch frequency | 0.218453 | 0.628967 |
| Original training-only item/video complete-watch rate | 0.217075 | 0.625639 |
| **Causally observed same-user prior-32 full-watch fraction** | **0.195408** | 0.591337 |
| Native TAC recurrent real history | 0.196248 | 0.578142 |
| **GRU recurrent real history** | **0.194484** | **0.574237** |

Native TAC Brier per paired seed: **0.195736 / 0.194474 / 0.198535**. GRU Brier: **0.193452 / 0.192996 / 0.197004**. GRU wins each paired seed and mean is **0.9% smaller** than TAC; history-only causal rate is also better than TAC on Brier, although TAC has lower log loss than that simple rate. This distinction matters; claims should be metric-specific.

**Memory-lesion test:** removing the entire actual prior32 interaction input while holding the current offered action constant:
- Native TAC history-blanked Brier **0.224182 / 0.238945 / 0.218800**; mean ~**0.227309** (compare native full-history mean **0.196248**).
- GRU history-blanked Brier **0.231609 / 0.235951 / 0.219007**; mean ~**0.228856** (compare GRU full-history mean **0.194484**).

The history dependence is real in both learned models, but does not establish a superior TAC encoding or counterfactual decision benefit. Blank input is a model perturbation, NOT an intervention in the human environment.

## Scientific decision and mathematically targeted next steps

- **PASS** genuine real logged actions and observed user responses, authentic source/manifest security, software causal-order checks, exact trainable parameter count and matched training-sample/optimizer budgets, saved auditable 3-seed original weights, and independent later-time evaluation.
- **FAIL** TAC superiority: stronger GRU and simpler causal running-user statistics beat it on heldout mean Brier. TAC does not justify its more complex and slower current recurrence on this task.
- **DO NOT CLAIM** equality of training FLOPs/wall-time, authenticated external ground truth, propensities, randomized unshown-item effects, improved recommendation policy reward, CASM action cost, persistent identity discovery, or System2 emergence. Observed platform video exposures are factual, but the model cannot identify the reward of actions not actually shown.
- **NEXT PREREGISTERED DIRECTION:** investigate future-utility-weighted retention credit assignment and entity-specific state write/read using naturally observed long histories, against a simple GRU and the actual user-history average, with matched parameters/data/compute; choose a dataset where genuine changes and delayed outcomes make retention mathematically necessary. Do not add decorative expert routers, another projection, or speculative oscillators before a specific observed failure in the retained state has been isolated.
- **RETAIN** all earlier ME001–ME013 original outcomes, unified master plan, CASM, CDL-PHR, executable memory, PNDS/Lean specification and hardware/complexity work; no SSM conversion or merge.
