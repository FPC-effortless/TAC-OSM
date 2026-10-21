# How the supplied review informs ME037

Verified PR300 remains open/draft/unmerged at d965964ad58f853465dda763e63106d9ea74caef.
Its committed results and source support the reported task-scoped TimesFM/TabFM
outcomes. This turn has not independently downloaded/replayed those foundation
checkpoints or their Actions artifacts. Its source report is preserved here.

The study does not establish a permanent domain boundary for TAC: it compared
specific small newly trained adapters with unmatched pretrained models and small
held-out cohorts. Neither does it demonstrate TAC's proposed action control,
persistent-lifetime state, microsecond streaming, VoI or causal repair advantage.
Those remain objectives requiring real action/outcome evidence. No new simulator
control study is substituted for such evidence. The tabular baseline is histogram
gradient boosting, not XGBoost, and native tabular training is not zero-shot.

ME036 already passed level/log-scale into the SwiGLU head. Its96inputs feed a
single state and direct horizon outputs, so a720-step forecast does not mean
720recurrent updates diluted that state. Patch stability and memory dilution
explanations remain hypotheses. ME037 instead builds and trains a more useful
full-window contextual representation alongside native memory, limits correction
amplitude in actual forecast units, and makes the forecast-lead response monotone.
No standalone attribution phase blocks construction.

The independent UCI household site provides new measured forecasts with train-only
preprocessing. It has no repair/action/cost labels and cannot qualify a controller.
The five previous benchmark test sets remain development-exposed evidence. Neither
this forecasting build nor PR300 warrants a general foundation-model ranking.
