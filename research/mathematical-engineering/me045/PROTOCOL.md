# ME045 preregistration — conditional multiscale TAC on genuinely observed physical measurements

Registered **before ANY new ME045 FIT or DEV scoring**. Stacked on ME044 PR309, which stays unmerged and unchanged. This is the user's architecture-first, complete-model training stage: combine ME043B conditional execution and ME044 orthogonal multiscale innovation decoder, benchmark a cheap GRU16 first alongside matched GRU40 and TAC memory, on two real measured domains. Do not waste runs on diagnostic-only component ablations.

## Architectural and mathematical treatment

All candidates receive a 96-observation real history X, run an identical, separately trained **frozen** full-history AR+context forecaster F0(X), and use the same known sampling cadence, exact 16-dimensional QR-orthogonal spline/harmonic forecast basis B, observed normalized 12x16 causal patch tokens, and four-feature split contextual latent from ME044. No new synthetic training or future measurements at inference. Native memory fast32 slow32 + 8x8 innovation matrix is unchanged. The native six-step update is unchanged, including context-seeded fast/slow initialization.

Three complete learned expert families:
- conditional_native: original ME044 native memory and 16-coefficient bounded residual decoder
- conditional_gru40: identical ME044 matched GRU40 recurrent encoder and *same* multiscale decoder (near-matched total active expert+router coefficients to native, document precise count)
- conditional_gru16: deliberately lighter GRU16 encoder, same basis and readout, **unmatched active parameters**, practical candidate required by user.
Each receives **the same active router**: 144 context/latent observed-only inputs ->16 SiLU ->1 logit, trainable with real future residual losses (train FIT only). No test labels, outcomes or gradients enter inference routing.

At inference, compute route pi_i = sigmoid(R_theta([c_i,l_i])) per observed channel/window. If pi_i <= 0.5, *do not execute expert for that channel*. If pi_i > 0.5, run only selected channels through native/GRU encoder and basis decoder; return cheap forecast plus bounded residual. For a selected channel:
    a_i = MLP_208to16([c_i,l_i,h_i])
    q_i = sigmoid(Linear_208to1([c_i,l_i,h_i]))
    d_i = 0.5 * sigma_i * q_i * tanh((a_i @ B)/0.5)
    yhat_i = F0(X)_i + d_i.
The gate is trained on improvement of the *true observed FIT outcome* (not synthetic counterfactual) from dense expensive expert evaluation:
    e0_i = mean_horizon [(y - F0)^2]
    e1_i = mean_horizon [(y - (F0+d))^2]
    z_i = 1[e0_i - e1_i > 0.01*e0_i]
    L_expert = MSE(F0+d,y) + 0.05*MSE(a,project((y-F0)/sigma)) +0.01*MSE(d,0)
    L_router = 0.05*BCEWithLogits(router_logit,z_i)
             + mean_i [ pi_i * stopgrad(e1_i - e0_i) + 0.001*pi_i ].
The expensive expert is evaluated DENSELY during training; native/GRU and gate train every adaptation update so there is **no optimistic conditional-training-compute claim**. During eval **real hard branch dispatch skips** unselected channels, not an after-the-fact probability mask. Report actual selected fraction by real channel/window; if gate never executes or always executes, report that instead of claiming intelligent routing.

## Identical authentic data, no prior-test reopening

ETTh1 official real hourly 7 numeric channels, original bytes SHA256 f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066; train indices [0,8640), historically viewed development [8640,11520). Tetouan official 10-min 8 numeric channels, original ZIP SHA256 3c4bf684161180937043a9fb65701a83d44b740b0d42f0492b6e7aec57ddbbfe, member SHA256 ca8b2f30cf07dd872cb7f2aa45dac110ee38dbe5fc5910a6ff4bcc530dae3a40; train indices [0,36691), previously viewed DEV [36691,41933). Full history 96 observed real rows, horizons 96 and 720 of **genuine recorded future outcome**. The official original test sections of both datasets remain unparsed in this study; they were exposed in earlier research and CANNOT serve as a new pristine sealed test. All mean/std fit on train ONLY. Primary ETTh1 **oil temperature OT** standardized MSE and secondary **all 7-channel** MSE. Primary Tetouan last three real power-zone standardized MSE, secondary **all 8-channel** MSE. Do not compare either to standard published multivariate ETTh1 test MSE (.45 or otherwise): different evaluated cohort, target-channel aggregation, and protocol.

## Registered 48 adaptation FITs plus eight common backbones

H ∈ {96,720}; seeds ∈ {45001,45002}; domains {etth1,tetouan}; adaptation arms 3; lanes fixed 1000 adaptation updates (batch4), or separate fixed10 optimizer CPU seconds (batch4, cap10000). Total 3 x2x2x2x2=**48 original trained conditional full-model experts**. Eight genuine train-only AR parents are each trained 500 updates with original same published source-only model/procedure and same historical-DEV parent selection100/250/500. Each domain/H/seed frozen context F0 is trained **1000 genuine original FIT-only updates**, original ME044 context objective/optimizer from that same original AR parent, without routing/expert and with same sampled FIT anchor sequence. The resulting frozen full-context checkpoint is EXACT source for all 3 candidate arms and both lanes in each group. Backbones never use original DEV outcomes to update gradients, but parent endpoint selection already uses historical DEV, and DEV repeatedly exposed.

Fixed-update lane exactly1000 expert/router AdamW steps, same true FIT sampled histories/targets in all 3 arms, LR.001, wd1e-4, clip expert separately router norm1, same frozen cheap coefficients, same seed+2000 sampling stream. CPU lane independently fit from exactly same pretrained frozen cheap checkpoints until10.0 optimizer-CPU-seconds, final update crossing allowed and counted, max10000. The same data sampler sequence starts in both lanes but exposes differing numbers of real examples by design. Record per-case exact param counts, expert/route gradient presence, parent/backbone cost separately, actual downstream optimizer time, overall process and wall time, RAM high water. No full training FLOP, wall/energy, or simultaneous equal-param/data/compute matching claim. GRU16 vs native never passed off as matched.

For every **single fixed trained checkpoint**, report a deterministic DEV inference comparison at route threshold0.5, force_always_on and force_cheap_only. These are operating modes of the *same learned model*, not extra trained arms. No post-DEV threshold selection; optional thresholds0.35/0.65 only reported on old DEV as diagnostic ROC-like rate/error curves, not substitutes for registered 0.5 scores. Report latency **batch1 full forecast**, 5 warmups + 5x20 repetitions per arm/mode/domain/H/seed/lane; same process CPU and frozen single observed DEV window; calibration of full-window latency is machine-specific. Hard latency target average native <=.45ms may be unrealistic; retain failure honestly. Record routed active fraction and per-horizon, per-seed paired MSE differences.

## Registered acceptance gates (historical DEV only, all compulsory)

- ETTh1 H720 conditional_native primary OT MSE **<0.190500** (user target), and paired conditional_native must beat always-GRU40 at **same actual inference latency or faster**; compare also with original ME044 GRU .191517. A similar fixed-CPU requirement compares fixed10s true optimizer costs.
- Tetouan mean fixed10s conditional_native primary power MSE **<0.101500** (user target), while beating matched GRU40 under fixed CPU.
- Native adaptive average batch1 full-model inference CPU latency **<0.450ms** on measured runner AND actual expert fraction <1 (not dense); routing must not trivially collapse to always cheap without measurable accuracy improvement against cheap-only.
- Joint cross-domain performance: conditional-native must beat its own cheap-only forecast AND conditional-GRU40 on registered mean original DEV error *in each domain* at both fixed-update and fixed-CPU training budgets. Report confidence intervals based on user-window/temporal block bootstrap when feasible; two seeds not independent real-world populations.
- All model weights, source, optimizer/RNG, true forecasts/truths, executed flags and timings replay according to pinned numerical profile; no original ME044 exceptions deleted.

These are ambitious gates not reasons to silently retune hyperparameters after seeing scores. If the native model fails **on a truly new sealed standard-protocol physical task at equal measured latency**, apply the user's stop rule: halt this native memory variant and carry the decoder into lighter backbones. Old exposed ETTh1/Tetouan development failures do NOT falsely satisfy that sealed stop precondition. Do not promote or merge any resulting model on the reused historical DEV alone.

## Cross-CPU numerical portability

ME044 original AVX512 to AVX2 bit-exact equality failed and STAYS FAILED. The original recorded maximal 80-model forecast drift was 4.76837158203125e-7 (not 1.14e-6 cited in the planning summary). Use pinned AVX2 profile ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 before Python import, 1 thread. Require original checkpoint-hash and model forecast exact replay in this pinned profile. If running on another ISA, keep original failure and independently validate a new registered float64 and forecast tolerance; do not relabel a tolerance pass as cross-ISA bit equality.

## Later agenda, not fabricated in this dataset

DeepLoop-scaled internal refinement, gradient-alignment measures, DeLoop-style real action utility/cost penalties, and executable CASM require separate real operational action logs and outcome/costs to support behavior/counterfactual claims. Do NOT assert these are finished because a forecasting expert was trained. We may pursue additional complete models without isolated component diagnostics after the current 48 model study is recorded.
