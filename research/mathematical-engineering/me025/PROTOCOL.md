# ME025 preregistration — real, causal, feature-parity native TAC versus a small Transformer

**Status:** registration and implementation stage, NO fitting or unseen outcomes opened at registration. **Do not claim an improvement or Transformer competitiveness until genuine out-of-population evidence is acquired.** ME024 original native equations and ME024 mathematically checked split RMS remain the benchmark. ME023 multi-depth causal supervision remains. No SSM conversion.

## Specific falsifiable question

ME021 showed authentic rate-EWMA features gave a tiny 225-param MLP far stronger performance than earlier unconditioned recurrent models. ME022–024 improved native TAC by readout conditioning, multi-depth learning, split RMS. Does giving native TAC **the same causal, fixed five-timescale watch summaries and genuine FIT-only target-item popularity** improve its real-event forecasts beyond unchanged ME024 RMS? Can a small, causal, parameter-near-matched Transformer use the same information more effectively on the exact same tasks? **Do not introduce a new memory mechanism or unrelated task.**

## Fixed architecture matrix

A. `tac_rms`: original ME024 `CoupledNative(coupling=0,rms_readout=True)`, exactly **135050 trainable params**, 64-d native state, RMS readout, duration FIT-standardized, causal prior32.

B. `tac_kernel`: exact same native recurrence/gates/states, **head only** enlarged by six input coefficients: five parameter-free causal normalized exponential summaries (half-lives **2,8,16,32,64**) + FIT-only smoothed offered-item watch popularity. Exactly **135056 trainables** (delta 0.0044%). Learned head remains initialized to zero, so initial forecast equals prior32. No new learned memory/SSM; these are observable causal rate features.

C. `tiny_transformer_kernel`: 2-layer, 2-head causal Transformer, d_model16/FFN48 with sinusoidal positional coding and feature projection to a 64-d split RMS readout; **134896 trainables**, within **0.12% of B**, same 16000×8 item embedding, known event metadata, same five causal rate summaries, same FIT item popularity and duration, same 32-event prior and **exact same four real prefix prediction targets**. Mask all future attention with fixed causal upper triangle; assert prefix noninterference. Extra attention cache / activations must be counted. Native TAC does not become a Transformer; C is only a challenger baseline. Transformer compute may exceed TAC considerably; do not claim compute parity from param matching.

D. `summary_mlp`: small fixed-feature tabular MLP trained on FIT and scored on same DEV; unmatched capacity practical reference, no param-architecture superiority claim. Same observed causal summary features and target-item popularity/duration. Prior-head guard preserved. Distinguish fixed summary from learned state.

The exact active parameter counts and feature shape must be checked by numerical preflight **before** fitting. These near-equal parameter numbers include the 128000 parameter learned item embedding. The 64d TAC state and O(T) inference vs O(T²) Transformer attention are materially different resources and must be reported.

## Actual data and exposure

Original publisher KuaiRec Zenodo 18164998 `big_matrix.csv`, source ZIP SHA256 `c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c`, MD5 `261550d472c48eff4990fb13c0e5bcf7`; Gao et al. CIKM 2022. Original `read_early()` rejects any outcome >= 2020-08-14 BEFORE label interpretation, original `original_128_windows()` 10,776 genuine FIT before Aug01 and 5,952 already-inspected DEV Aug01–13, user ID%16==0. No new user4/5 first-exposure claim, no MetroPT fresh-week claim and no other reserved cohort. Every 128-element history contains genuine prior events. At prefix k the model sees the k prior actual observations, **not any feedback at event k**, with offered item and its pre-response duration observed; summary weights normalize strictly over observed events before the prediction boundary. Item-popularity counts use only FIT labels.

All neural arms receive same fitted FIXED itempop tables and user-rate kernels. No one receives unique privileged target features; Transformer is causally masked. The standalone summary MLP has no item-identity learned embedding and is identified as practical but information-representation-unmatched.

For each of seeds 14001–14003, exactly 300 AdamW optimizer steps × batch 128, identical seeded shuffled FIT target window index stream, all arms observe exactly 128 genuine past events/window and four **real** next-event labels/window. Every objective: 1/6 BCE at k=32,64,96 and 1/2 BCE at k=128. Same clipping 1, decay 1e-4, body LR .001, head LR .006, same machine/tuning opportunities; log dense steps 0/1/25/50/100/150/200/250/300. Fixed upper per-fit CPU budget 600s. If an arm cannot finish at both matched steps AND available CPU/FLOPs, mark the three-way parity gate FAIL rather than adjusting the budget or hiding incomplete fits. Actual wall/CPU seconds and peak RSS per fitted arm, model params, deterministic data draws, scores, logloss, AUC, ECE. FLOPs not fully comparable unless measured with identical profiler; exact total compute parity is **NOT** established by matched steps, ceilings or observed CPU alone.

## Endpoint, hypothesis and decisions

All scoring occurs ONLY on the **already-exposed historical DEV** and is exploratory engineering evidence. Primary (development-only): `tac_kernel` minus `tac_rms` final mean Brier; pass candidate gate iff all 3 per-seed differences negative AND mean <= -0.0010 and ECE doesn't worsen >0.005. Secondary descriptive: Transformer vs TAC_kernel (plus MLP), paired per-event scores and resource accounting. No significance, no independent architectural ranking or capacity-superiority claim from historical DEV and uncontrolled FLOPs. Do not tune after first scores. Preserve negatives. If kernel candidate loses, reject and keep ME024 native unchanged. If it wins, still require genuine **new** reserved out-of-user/out-of-time population before promotion, and preferably train/eval on external public real data with similar temporal response tasks. Do NOT describe ME019F user4 (already opened) or ME021 user5 (already opened) as pristine or independent.

## Audits

Preflight only tests constructed input mathematical software properties—initial logits equal genuine prior, exact coefficient/kernel formula, registered active parameter gap <=1%, causal strict prefix noninterference at 32/64/96, no logits dependence on unobserved current outcome, finite gradients, strict streaming state+kernel update equivalence on chunked inputs, and observed input metadata shape. Artificial inputs are NEVER performance evidence. Real phase must hash publisher source and preserve genuine per-event probability outputs, all checkpoint digests and run logs. The existing 897 repository tests must remain green.

**Important limitation:** even when a Transformer beats native on reused DEV, this is NOT proof of Transformer competitiveness in the full parity sense; and even if native wins, do not claim beating Transformers. Preserve the independent original ME024 research result and do not merge or deploy.
