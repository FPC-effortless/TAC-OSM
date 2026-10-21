# ME041 registered multi-architecture model search — real-world evidence and full system performance

2026-10-21. Stacked draft on original ME040, before any new ME041 model training. The user requests **broad architecture-first development** testing multiple mathematically distinct master-plan ideas *in trained complete predictive models* rather than stopping after one tweak or returning to diagnostic-only experiments. Preserve previous results. No synthetic evaluation success.

## Starting point and evidence

ME040 exact same original native weights and fixed paired six-update path accelerated native training 1.604x (equal 250 updates), preserved Tetouan power MSE but **failed equal five-CPU-second context-model competitiveness**: paired 0.118321, original native 0.119841, context 0.104675 (two registered horizons, two seeds). ME038 useful associative delta-rule memory and end-to-end context/AR remain the base. Do NOT claim ME040 native architecture qualified or ignore its cheaper context control.

## Native mathematical family — eight genuinely distinct **trained** models, not eight ablations

All eight use the same full original source architecture (AR backbone, 96-observation contextual readout, channel-coupled patches, fast32/slow32 nonlinear recurrence, 8x8 prediction-error memory) and its exact 6 native updates on consecutive **already observed** patch pairs. Every model adds an **identically sized, active 16->16 affine adapter (272 trainable coefficients)**, also using precisely the same tensor shapes and the same original AR parent/initial forecast, and the same scalar training loss. Each strategy interprets the adapter outputs differently:

1. baseline_adaptive: tiny affine temporal adapter (small bounded residual at every paired token), practical capacity-matched base.
2. learned_relevance: input-conditional convex aggregation from two observed patches; adaptively chooses which past segment to emphasize.
3. cdl_plasticity: learned positive relevance from past-only u modulates the **prediction-error write coefficient** of the same exact associative 8x8 memory; no causal counterfactual claim.
4. sparse_assemblies: four groups of four adapter-produced features with normalized sparse top-two group routing, applied as a bounded input innovation; software sparsity does not prove emergent entities.
5. oscillatory_phase: smooth patch-index-dependent sine phase modulation of the adapter and causal observed tokens; this is not a fully phase-coherent NADOC model or proven neuroscience.
6. role_filler: learned signed multiplicative binding with an index-rolled historical patch; a differentiable VSA-inspired binding hypothesis, not established symbolic composition.
7. latent_refinement: two bounded **internal latent readout refinement** iterations after the world state has been updated six times; internal iterations get no additional external observations or executed actions; does not prove System 2 or CASM planning.
8. unified_multimechanism: reuses the **same 272 trainables** to combine adaptive pairing, relevance-dependent matrix write, sparse assemblies, phase modulation, role binding and two internal readout refinement iterations. All mathematics are original native operations composed; no new attention, global planner, SSM, Transformer or language decoder.

Also train **unchanged ME038 12-update native**, **unchanged ME040 six-update native**, and **ME038 strong context-only** as honest practical references (these have fewer active parameters than the eight added-adapter variants; the gap is <1% of full trainables, but this is still NOT exact full resource parity and may conceal capacity in the small memory core). This is 11 arms total.

## Native update math

The original delta memory update is **unchanged** except cdl/unified relevance scales beta in [0,1]:
    k_t = normalize(W_k[u_t,f_t]), v_t=tanh(W_v[f_t,s_t])
    M' = rho M + beta_t(v_t-rho M k_t) k_t^T

For learned aggregation let a=adapter(u_new-u_old), weight λ=sigmoid(mean(a)), and u_pair=(1-λ)u_old+λu_new. For fixed-pair paths use u_pair=(u_old+3u_new)/4. Any correction to u is bounded by tanh/adaptive scaling; nonzero gradients must reach the original native recurrence and every new adapter matrix. Sparse group intervention selects the top two of four group scores; learning the affine weights remains differentiable through selected gates. Phase modulation uses only patch index j=0..5 plus observed features, never future data. Binding uses a deterministic circular coordinate permutation multiplied by learned role activations. Latent refinement runs twice on a separate readout copy of the learned h=[f,s] without rewriting f,s,M or creating fake observations. All predictions are defined only after the full 96 observed samples.

**No invented new machine/control outcomes, no policy/counterfactual/Sys2/learned-operator claims.** Entity-discovery, actual counterfactual retention, real action-conditioned CASM, verified consolidation, and multimodal transfer **need their own real-world datasets and tasks**, and this six-step *forecasting* suite does NOT magically qualify them. These will be carried into subsequent genuine complete-model evaluation with authentic action/event/cost data rather than fabricated labels. The eight new arms are practical, first trained implementations of selected mathematical ideas, not full tests of every ambitious master-plan component.

## Real source, exposure and test

Original UCI849 publisher Tetouan power/weather CSV 52,416 10-minute observations, exact ZIP SHA256 3c4bf684161180937043a9fb65701a83d44b740b0d42f0492b6e7aec57ddbbfe and member SHA256 ca8b2f30cf07dd872cb7f2aa45dac110ee38dbe5fc5910a6ff4bcc530dae3a40. Original FIT numeric rows [0,36691), historical repeatedly examined DEV [36691,41933). Numeric values for [41933,52416) not parsed here. The final20% was **ALREADY OPENED BY ME038**, cannot be described as a pristine reserved new test. All forecasts are on genuine measured values; eight real channels, primary three measured power zones, 96 observed ten-minute values, horizons 96 and 720; no future power/weather/calendar inputs. Source, original column order/cadence and only train-fitted normalization verified.

## Registered training/evaluation

Same source-only four independently trained AR parents H∈{96,720} × seeds {41001,41002}, 500 updates each, batch4, select only from development endpoint steps {100,250,500}. Parent costs charged to complete pipeline. 11 complete-model arms × 2 horizons × 2 seeds × 2 lanes = **88 genuine complete neural fits**; each separately initialized from matching parent, original native/head/AR code, zero correction initial logits, no result-driven retuning:

- Equal-exposure lane: 250 genuine downstream optimizer steps, batch4, exactly the same sampled anchor indexes and targets per H and seed, original objective MSE + 10*AR drift anchor + .01*bounded correction², AdamW AR lr 1e-4, remainder lr 1e-3, wd 1e-4, grad clip 1, fixed seed+2000. H96/H720 real measured target sequences. Entire step250 chosen for primary development scoring; dense final/full trace and step100 training receipt.
- Equal-CPU lane: separately initialized, same parent/seed; ~5 process CPU-seconds of optimizer work (finishes crossing step), max4000 updates. Report actual examples seen and unequal exposure. No claim of equal training FLOPs/energy or simultaneous exposure and CPU parity; do not overtrain a cheaper model simply to match slow model data.

Active parameters across model families within 1% total parameter budget for each horizon; eight experimental adapter arms match EXACT trainable count, input data, original AR and exposure. Non-adapter references within 1% but not exact. No dummy weight padding, adapter grads and native grads nonzero.

## Registered performance selection rules

Historical DEV only, no independent test:
- Rank *all 11 arms* by mean power-zone standardized MSE over the **four H×seed cases at registered step250**; display per horizon, seed and real CPU cost. Candidate must improve mean MSE over original native and context at step250, and NOT be worse than original native by >5% at either horizon. Separately require equal-CPU MSE <= the context baseline on average before calling it **CPU competitive**. Record a Pareto frontier (accuracy vs actual optimizer CPU) including context and all native arms. No architecture promoted/merged and no Transformer-superiority claims from reused DEV.
- Explicit one-time development ranking is an exploratory search over eleven arms; winning model is statistically selected on old DEV and requires *new* physical-system qualification. Even if best appears strong, do not claim held-out accuracy, learned decision utility or fair complete-pipeline FLOP parity. Honest scoreboard includes EVERY original trained arm and failed run; all checkpoints/optimizer/RNG and SHA originals remain available.

## Verification and operational limit

Numerical-only constructed-input tests enforce original exact 8x8 innovation operator on baseline/role/phase/sparse/etc, chosen CDL gate stays inside [0,1], finite full loss and gradients through all trainable parameter families, causal patches, identical source-only parent initial forecasts, fixed per-model active counts and no unavailable data. Synthetic tensors are *software unit tests only*. Every genuine development model's original saved predictions/checkpoints must replay within tolerance; report exactness separately, do not quietly retrain. CI archives all fits, failures, source receipts, resource records and code. Up to 110min single-core GitHub Actions with CPU/staging; run fully in this turn if feasible. Do not report success while any gates or fits are pending.
