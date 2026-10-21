# ME047 — Capability-first TAC: transferable observed signals, typed goals, slow macro guidance

**Preregistered 2026-10-21 BEFORE any ME047 training.** Stacked on ME046 PR311, preserving all earlier negative gates and unmerged checkpoints. This is capability-first, not an edge inference speed project. Real-world training/evaluation and compulsory honest baseline comparisons remain in force. Constructed/synthetic data ONLY for numerical unit tests, never success claims.

## Hypothesis and actual learned functions

A single pretrained shared 1-channel temporal tokenizer learned on two authentic physical source datasets may help TAC-OSM learn forecasts and goal-targeted outcomes that TRANSFER without gradient updates to another recorded transformer site. The proposed native intelligence improvement combines: (a) a stable frozen cross-domain observation encoder, (b) explicitly typed goal/observable metadata, and (c) a frozen heavy macro forecast supplied as an anchor to TAC's slow memory. This is NOT arbitrary English interpretation, universal zero shot reasoning or a trained maintenance/advertising control policy. Real-action learning requires a different logged-interventions/outcomes domain.

### Frozen source-trained tokenizer

Tokenizer: Conv1d(1,24,5,pad2), GELU, Conv1d(24,32,5,pad2), GELU, 8-step mean-pool -> 12 tokens of 32 values for a 96-observation true history. Pretrain 600 updates batch32 across both publisher-authentic source FITs only to predict the **last observed 8 measurements** using preceding 88 source observations, normalized by their own preceding 88. It is valid real masked temporal self-supervision, not synthetic/preprogrammed future. Save weights+optimizer/checksum. Freeze token coefficients for ALL complete-model arms.

### Frozen source-trained macro teacher

Independent Linear(96,8) teacher, trained 600 updates batch32 on the same two sources, predicts eight genuine future-bin means over H96 from real observed history and true later measured targets. At model inference it sees ONLY 96 already observed measurements. Can be run out of band and passed as cached macro_anchor input to the predictor; if unavailable, predict locally from previous history (not a proven active asynchronous infrastructure). Save teacher weights+optimizer+SHA. Slow native initial state receives s0=0.1*tanh(W_macro(macro_anchor)); actual six-step fast/slow+8x8 associative innovation recurrence remains original ME038/040.

### Typed semantic goals; no pretend SLM

Gateway takes validated structured {goal,role,threshold} metadata. Goals=mean,peak,last of the actually recorded next96 target points; source channel role=unknown,load,temperature,environment,power. Typed metadata encoded in a learned 32D goal embedding and 8D role embedding. Candidate world state + predicted macro8 + intent32 + role8 feeds a goal prediction head; native/context/GRU/Transformer also predict the actual entire 96-step trajectory. The scalar goals are genuine derived statistics of measured future, not imaginary labels. Hard numeric action caps are validated and tested as a deterministic software envelope only; no operational intervention success claim.

### Four fully trained predictor architectures

- hybrid_tac: same fast32/slow32+8x8 associative delta-rule native recurrence over six paired learned-tokens (32->16), macro-seeded slow state, trainable goal+trajectory heads.
- hybrid_gru: GRU(32→64) sequence state with macro seeded initial hidden, identical frozen tokenizer/macro and same heads.
- hybrid_attention: two causal 4-head 32D Transformer encoder layers over same observed token sequence, same macro/goal heads, not an LLM or externally pretrained Transformer.
- hybrid_context: nonrecurrent context MLP over all observed 12x32 tokens, same macro/goal heads.

Each model has genuinely active different parameter counts, with exact counts/time in results. Shared trained tokenizer and macro teacher have identical frozen weights. No false parameter/FLOP/time-matched architecture-superiority statements; same data, sampled windows and optimizer updates, honest disparity logs. Compare persistence baseline trained on neither dataset.

## Authentic data and transfer boundary

Source1 ETTh1 original hourly first transformer SHA256 f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066. True source numeric FIT rows [0,8640), old DEV [8640,11520); original benchmark test not used.
Source2 Tetouan official 10-minute weather/power, ZIP SHA256 3c4bf684161180937043a9fb65701a83d44b740b0d42f0492b6e7aec57ddbbfe, CSV SHA256 ca8b2f30cf07dd872cb7f2aa45dac110ee38dbe5fc5910a6ff4bcc530dae3a40. Real FIT rows [0,36691), old DEV [36691,41933); former final20% remains unopened here. Train only on FIT, development outcomes historically reused.
UNSEEN TARGET: ETTm2, second transformer region 15-minute actual hourly transformer family different physical station. Pinned original GitHub source commit 1d16c8f4f943005d613b5bc962e9eeb06058cf07, path ETT-small/ETTm2.csv, 69,680 real rows, Git blob SHA1 f9d46027c987d442d2d8f81308d1d8c4d1d35db8, 9,677,236 raw bytes, original URL https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTm2.csv. Compute/record additional raw SHA256 at first fetch, verify Git blob SHA1 before parsing. Critically **NO ETTm2 data loaded until AFTER all shared pretraining AND original 8 downstream models are frozen, SHA-saved**. Prospective held-out target origins [48096,52096) stride96, real H96 outcomes; one opening, no adaptation, no per-target fitted mean/std, only causal per-window x mean/std. The station is different but belongs to the ETT family, so successful transfer here does not prove arbitrary cross-domain universality.

## Registered training and comparison

H96 throughout, two seeds 47001 and 47002, four complete models per seed = EIGHT authentic downstream fits; 1000 AdamW updates per fit, batch32 true 96 history / next96 FUTURE measurements, equal exact input samples source domain and channel by seeded sampler, goal sampled among three by same seeded generator, LR1e-3 wd1e-4 grad clip1. Loss = MSE(96-step forecast normalized using observed-only history mean/std, measured 96-step normalized future) +0.3 MSE(goal scalar forecast, actual future goal) +0.15 MSE(8 forecast-bin means, actual future bin means). No synthetic forecasts used as targets. Shared encoder & teacher pretrain 600 real updates each; freeze for all. Log optimizer CPU and examples, active parameters, checkpoint and RNG checksums.

Score SOURCE DEV without selecting checkpoints or retuning; only after all 8 finished, open untouched target station ONCE and evaluate all arms+observed persistence on same 42 windows times seven channels. Report normalized trajectory MSE/MAE, goal-specific MSE+MAE for 3 intent types, actual macro prediction MSE and visible per-station numbers. If native does not beat context+GRU+persistence on target trajectory AND all three goals, **fail** capability superiority gate. True zero-gradient transfer may be achieved even if not competitive; define it operationally as a frozen checkpoint executing on unknown site's real inputs without retraining, distinct from beating control baselines. Also record mean and peak target calibration, no semantic English or device speed claim. Final original test source values never viewed by training.

## Preserve capability constraints

Schema-validated goals and independent hard cap checks do not qualify safe autonomous execution. No language parser, SLM, Needle checkpoint, System 2 cloud computer, policy learning or real external interventions are trained here. Genuine action-conditioned forecasts and decisions must be evaluated next on real logged action choices, propensities and verified outcomes (for example ZOZOTOWN Open Bandit Dataset); observed interactions alone cannot establish causal interventions without assumptions. Original ME044 AVX512 vs AVX2 exactness failure stays documented; run pinned AVX2 replay and preserve every failure. No branch merge/promotion based solely on single ETTm2 target site.
