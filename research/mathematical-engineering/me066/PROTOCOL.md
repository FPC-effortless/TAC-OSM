# ME066 — four real-trained compact world models, prospective source registration

## Status at registration
- Work is stacked on [ME065 draft PR #344](https://github.com/FPC-effortless/TAC-OSM/pull/344). Its four-arm original-FIT preflight was a genuine 160-episode *untrained* source-only test, and passed numeric parity: active parameters **37,177 / 37,431 / 37,319 / 37,415**; measured operator FLOPs/update **56,895,406 / 56,531,772 / 56,423,631 / 56,841,027** for native TAC, GRU, compact RSSM, diagonal selective SSM. Max difference 0.679% parameters, 0.829% profiled FLOPs. CPU equality not required.
- This stage preregisters full real-world training under identical original publisher FIT inputs, original task schedules, genuine measured optimizer updates and compute ceilings. **A registered manifest does not itself constitute a completed model fit or positive result.**
- ME064 already exposed spray-nozzle target; it is ineligible for new independent qualification. A *different* task from metadata-only still-unopened publisher candidates must be selected irrevocably, and original trajectory files cannot be touched until all original model bytes exist in Git.

## Compulsory preregistered design
Training source: pinned original BodenAI RW-RL-HIL five-task FIT, 160 source episodes, identical 14D action/state features, human-override censoring, 64-frame causal history and 16 next-observation horizons. FIT original hash 4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776. Source task CAL remains completely excluded from fits, model selection and normalization. Original old TEST stays sealed.

Four arms: Native TAC ME063 (fast/slow/associative), best existing matched GRU, compact Gaussian RSSM, selective diagonal SSM (control only, not full Mamba). Share original physical persistence, FIT servo, 3-way causal regime router, output head, action information and physical MAE/Gaussian objective; RSSM's 0.005 KL is a transparent additional variational training objective rather than an assertion of isolated memory causation.

Five new seeds **65001–65005**, exactly 600 AdamW updates per arm/seed, 8 factual samples/update, exact identical original episodes/sample windows/order per paired seed, same optimizer schedule and no heldout-driven tuning. Pre-registered 20 original full model/optimizer/RNG checkpoints; no score permitted until all original bytes/checksums and code, dependencies, sampled schedule, publisher manifest and clean-independent model restoration are committed in Git.

Stage order is enforceable: source-only prefit PASS -> source FIT and new heldout publisher METADATA-ONLY registration -> immutable FAIR_COMPARISON.json with numeric prior measurements PASS -> checkpoint byte-storage proof -> FIT optimizer 4×5 training -> final actual profiled parameter/data/compute audit PASS -> commit trained original checkpoint bytes -> independent checkout original checkpoint hash/load verification -> original source CAL and exactly one previously untouched new-task evaluation -> independent numerical score replay with explicitly justified numerical tolerance and exact nonnumeric identity. No implicit scoring workflow dispatch or target opening before checkpoint freeze.

Primary gate on new original heldout physical actions: TAC >=2% lower MAE than **all** of GRU/RSSM/selective SSM/last-state persistence/FIT servo, paired 4-episode-block CI upper bound <0 and meaningful independent seed stability. Evaluate per-horizon false physical movement, task transfer, source-only interval coverage+sharpness. Calibrated abstention, policy utility and causal alternative-action skill are NOT inferred merely from observed factual motion trajectories. No Woyengi autonomous effect and no claim of universal task generalization.

All negative source preflights and training attempts/checkpoints (if fitted) remain retained. If any parameter/FLOP/data parity or source integrity gate fails, do not silently relax the budget or score a known target. No synthetic/simulated performance claims, no equal CPU wall-time mandate, no dummy parameter padding.

Registration source: `scripts/me066_register_real_fit.py`.
Metadata record and manifest (only after GitHub source registration): `research/mathematical-engineering/me066/{TASK_LEVEL_SPLIT.json,FAIR_COMPARISON.json}`.
Real source parser inherited unchanged from `scripts/me064_mixed_real_fit.py`.
