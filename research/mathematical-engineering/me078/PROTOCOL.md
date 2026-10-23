# ME078 — integrated native kinematic multi-horizon prediction

Prospective real-data registration. **15 original fits**, three architectures ×
five paired seeds78001–78005. Complete native fast32/slow32/8×8 associative TAC,
GRU56 and two-layer causal RMSNorm Transformer32/4heads. All share the same
encoders, input information, kinematic forecasting interface, ordered quantile
heads, motion-event supervision and training protocol. Width differences only
match actual useful parameter/operation budgets, never dummy padding.

Source: original CC-BY-4.0 BodenAI RW-RL-HIL pinned publisher revision, exact
160episode SHA from ME076/077. Same120FIT/40exposedREPORT original episode split,
FIT-only scales/servo/coordinate conversion. No CAL, protected tissue-box or
sealed TEST access. This is **exposed DEV**, not a prospective new-site claim.

Inputs: eight observed14D states, lag1known logged commands transformed to the
same state coordinates, observed velocity/acceleration and nominal1/15s time.
The output at each prefix sees only that prefix; attention is causally masked.
No future logged actions or human takeover as model features. Dense supervision
at all eight prefixes for actual states1/4/16frames later. Unobserved futures or
post-takeover targets are masked out explicitly, never treated as persistence or
successful control. Same targets/masks and sample order for every architecture.

2048AdamW updates, batch24,lr0.0007,wd0.0001,clip1.0, final checkpoint only. No
warm starts, model selection, restarts or hyperparameter search. Exact training
order is frozen perseed. Equal-task windows use half uniform/half actual moving
samples, with **exact mixture-proposal importance weights** returning the natural
within-task objective. Loss: physical-unit pinball at .05/.5/.95, normalized by
FIT-only typical displacement and horizon, plus0.1weighted event BCE. The
separate global event probability is **not claimed to be implied by the quantile
endpoints**; quantile order alone is not a coherent full joint distribution.

Preflight numerical operator profiles: all floating ATen ops must be classified;
include forward/backward, gradient clipping and an analytical AdamW charge.
This is a documented **algorithmic scalar-operation estimate**, not actual
hardware FLOPs/instructions/energy. Matched ratios <=1% in this convention are
compulsory along with <=1%parameters and identical exposure. Measure the fitted
model with the same actual first FIT batch; graph/resource mismatch blocks final
qualification. Never inflate work or discard unsupported operator costs.
The initial untrained GRU decoder width122 was raised to123 solely to meet
resource parity before fitting; no predictive evaluation or tuning informed it.

All15whole-model original checkpoint/optimizer/RNG/live receipts must be frozen
in Git BEFORE REPORT scoring. Score the exact960ME077 query origins at H1;
report actual future-eligible denominators at H4/H16. Keep raw predictions,
targets, intervals, fixed event probabilities and all physical references. Restore
and rescore all15originals from a separate clean checkout. Raw replay tolerance
rtol1e-5/atol1e-6 fixed before fitting. No original result/checkpoint overwrites.

## Registered utility and comparison gates

Native H1all MAE >=5%below both persistence and frozen ME077; native H1moving
MAE >=5%below the **best** persistence/servo/clipped-velocity physical reference;
no native task >10%worse than persistence. For a developmental architecture win:
>=2%lower aggregate horizon-relative-to-persistence MAE than BOTH equally
resourced new GRU and Transformer, >=4/5seed wins against each, paired episode
97.5%CI upper(native-control)<0 for both (Bonferroni family95%).5000resamples,
seed78078, averaging model seeds per factual query before resampling40episodes.
Uncertainty conditions on fitted models and exposed source; no independent
intelligence or general architecture ranking even on success.

Report global event Brier, AP/AUC, fixed-threshold .1/.5 precision/recall and
top10%capture, overall/moving/stationary MAE, physical interval coverage/width/
proper interval score, and all tasks/horizons/seeds. No picking a favorable
horizon, best seed or retrospective oracle router. Report negative gates intact.

## Capability-to-Value Card

Skill: predict real evolving physical states over short trajectories, avoid
false changes and forecast actual movement. Potential advisory operator question:
is a system likely to move and where will it be within about one second?
Customer/operator usefulness UNKNOWN/NOT_MEASURED. Source autonomous-transition
censoring and logged commands cannot establish causal intervention value,
closed-loop control, risk safety, permission or policy quality. Woyengi authority
stays external. Original Git replay unlocks research evidence only, no business,
funding, deployment or commercial readiness. This physical lane does not qualify
enterprise workflow forecasts or long-lived entity memory.

Prior art: Gated Delta Networks, GRU (Cho et al.,2014), causal attention
(Vaswani et al.,2017), quantile regression (Koenker & Bassett,1978), and importance
sampling. Established mechanisms are reused and credited. The new question is
whole native predictive capability on authentic data, not attribution.
