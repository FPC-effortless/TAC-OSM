# ME040 — dual-rate coupled native innovation memory, end-to-end real performance and cost

Registered 2026-10-21 before new fitting. Research-only draft stacked on ME039. Architecture-first: build and train better full models; defer mechanistic ablations. Retain the compulsory authentic-real-outcome, parameter/data/compute equality and negative-result rules.

## Motivation and architectural change

ME038 innovation TAC (fast32, slow32, 8x8 error-correcting associative matrix per channel) reached a Tetouan new-period power MSE of 0.153310 under 500 matched updates versus context 0.156326, but lost under a 5 CPU-second optimization budget (native .151090 vs context .146461). ME039 TorchScript only sped training 1.076x; its training gate failed. Aim to reduce the number of sequential expensive memory writes in the actually trained TAC model, NOT replace it with a Transformer, SSM or planner.

ME038 processes 12 consecutive observed eight-step patches (8 local and 8 same-time shared cross-channel readings) with twelve applications of its coupled nonlinear F_TAC(f,s,M,u) transition. New paired_innovation_tac uses EXACT same native weights, state, associative update rule, learned gates, channel aggregation and full 96-observation context/AR/readout. For j=0..5 define the 16-feature causal paired token

    u*_j = (u_2j + 3*u_2j+1)/4
         = 0.5*(u_2j+u_2j+1)+0.25*(u_2j+1-u_2j)

and perform one complete coupled F_TAC transition for every u*_j, six transitions total. This fixed local quadrature emphasizes the more recent observed patch, preserves constant patches and sees all 96 observations. It deliberately loses detail: NOT equivalent to 12 transitions, not an associative scan, no claim of equal accuracy. The complete AR, normalization and context predictor still see all observations. Both native architectures have exactly the same trainable parameter count and 128 float32 recurrent values per channel: fast32, slow32, matrix64. Bounded fast/slow recurrence and conditional matrix operator bounds from ME038 apply; no global contraction claim.

## Authentic dataset

Only original real publisher UCI849 Tetouan City power/weather 2017 10-minute records, official zip SHA256 3c4bf684161180937043a9fb65701a83d44b740b0d42f0492b6e7aec57ddbbfe and CSV SHA256 ca8b2f30cf07dd872cb7f2aa45dac110ee38dbe5fc5910a6ff4bcc530dae3a40. Verify source exact hashes, 52,416 rows, eight numeric channels, unbroken 600-second cadence and original column order. Parse outcome values only in rows [0,41933). Train [0,36691), validation [36691,41933), all original last20% records [41933,52416) remain unexamined for ME040, though ALREADY EXPOSED by ME038. Original full-window 96 observations, channel-specific train-only normalization, forecast all eight measurements; separately evaluate the last three real zone power columns. H=96 and H=720 (10-minute steps), no future weather, invented outcomes, artificial training or policy labels. This is engineering on repeatedly seen historical validation, NOT fresh independent proof.

## Fixed experimental design

Train four AR-only parents: two horizons by seeds 40001 and 40002, authentic FIT, 500 AdamW updates, batch4, fixed generator(seed+1000), LR .001, decay1e-4, clipping1. Checkpoints at100/250/500; select lowest validation MSE. Each trained neural arm initializes from exactly the same source/horizon/seed AR parent and passes bit-exact parent-output equality with its zero-initialized correction.

Three complete jointly trainable models: innovation_tac (12-step unchanged), paired_innovation_tac (6-step same weights), context_cross (no native memory, strong original contextual competitor). Two horizons x two seeds x three arms x two independently trained lanes: 24 total adaptation fits.

Exposure lane: exactly 250 updates batch4, identical samples (sampler seed+2000), same AR parent, same AdamW LR AR .0001 vs rest .001, weight decay1e-4, clip1, loss forecast MSE+10*AR anchor+.01*correction square. Save step100 and250, compare fixed step250 (no DEV checkpoint cherry-picking).

CPU lane: separately initialized from identical parents and seed streams; spend 5 process-CPU seconds on optimization, finish crossing step; max4000 steps; actual updates and unequal target presentations MUST be reported. Exclude startup, validation, persistence and serialization from optimizer CPU but report them separately; actual total compute is unmatched. Score registered final step of each lane. Both lanes preserve original checkpoints, optimizer and sampler RNG, time/peak RSS, validation predictions and per-channel MSE/MAE. No changes after first outcomes.

Registered engineering gates, historical DEV only:
- SPEED: paired/native geometric mean ratio of charged 250-update native CPU >=1.35 across four matched pairs.
- ACCURACY: paired power-only mean MSE <=1.02 times original native AND <=1.02 times context; no per-horizon mean >1.05 times original native.
- FIXED_CPU: paired mean power MSE <= context AND <= original native when both trained to ~5 optimization CPU seconds, comparing practical capability, NOT fair training-data parity.
Report all 8-channel scores, full per-horizon scores, negative gates, per-seed accuracy and CPU. Parameters <=1% gap across paired/native/context, no inactive matching weights. Same source/training examples/selection privileges for exposure lane, different CPU/FLOPs; same target CPU limit but DIFFERENT training-data exposure in CPU lane. Strict architecture superiority and fully fair comparison claims FORBIDDEN. Full process energy/FLOPs and pretraining parity not proved.

Numerical-only software tests: same parent outputs initial, deterministic convex 2-patch aggregation, equivalent native parameter/memory sizes, true gradients after initial correction, PyTorch scripting, no future patch leakage, no leakage from held-out numeric rows. These are never performance evidence. Preserve original ME038/039 model files and all failed receipts. Publish complete attempts/weights/probability-equivalent real forecast arrays on GitHub Actions. No promotion, deployment or merge.

If the paired model fails, retain ME038 original models and no post-result parameter/tolerance adjustments. If it passes, next phase strengthens within-pair information extraction and qualifies on a different new physical task before a broader competitive claim. Do not hold further architectural improvements hostage to diagnostics, but maintain authenticity and strong baseline controls.
