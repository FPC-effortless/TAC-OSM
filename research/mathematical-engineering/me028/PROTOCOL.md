# ME028 registration — train the ME027 native nonlinear readout, no recurrence change

## Scientific motivation / source lineage

ME027 (draft PR #289) completed 12 fits on authentic KuaiRec, and failed its state reliance gate. On 300 fixed matched example-updates, 252-unit feature-MLP Brier .165386 vs native TAC linear head .166406; at 50 updates MLP .164508 vs TAC .186103. Fixed-CPU forced the faster MLP to train for ~3477 steps and overfit, so did NOT demonstrate superiority of native learned state. Original memory dynamics and learned state value remain unqualified. Read its ME027/MATHEMATICS.md and ARCHITECTURE-DIRECTION.md before changing architecture.

Netflix GenRec suggests predicting desired REAL user outcomes from richer context without text generation; NCM coarse-to-fine and Sep-Latent suggest useful intermediate representations should earn value via actual downstream forecasts; Hybrid Reasoning Agents separates hidden computation from external action. These are **motivations not capabilities imported into TAC**. This experiment tests ONLY whether a tiny nonlinear observable correction makes existing native state more useful; it does NOT add latent reasoning steps, separator tokens, attention, hierarchical memory, policy actions, or an SSM recurrence.

## Registered architecture: exact 310-param head correction

Use ME027 real-data observable 26-vector x (offered-item embedding 8, FIT-normalized known duration 1, five strictly past EWMAs, FIT-only popularity 1, mean 11-d past event encoding). Original native 64-d fast32/slow32 recurrence and split RMS state φ(h) **unchanged** (no slow->fast feedback). Analytic prior32 is unchanged. The new target logit is

    ℓ = logit((1 + Σ(last 32 *observed past* full-watch))/34)
        + w_hᵀ φ(h) + vᵀ tanh(A x + a) + b

A: (12,26), a: (12,), v: (12,), w_h: (64,), b: scalar. All 401 readout trainables vs original 91, **310 extra**. Whole model 135377 parameters vs 135067 original (0.230% relative). Head w_h,v,b are exactly zero initialized, so initial forecast equals prior32. A receives standard, seeded PyTorch Linear initialization, never changed after seeing any dev results. The nonlinear correction is **visible in pre-response observables**; it is not extra learned recurrent memory and is not supposed to be called learned reasoning.

At fixed other inputs x, derivative of logit with respect to stored standardized duration d:
    dℓ/dd = Σ_j v_j (1-tanh(z_j)²) A[j,d]
    d²ℓ/dd² = -2Σ_j v_j tanh(z_j) (1-tanh(z_j)²) A[j,d]².
These derivatives are **with respect to standardized stored duration**, not raw seconds; memory is computed from past-only history and does not depend on offered/current duration. Head derivative tests compare autograd against this exact expression. No bounded global sensitivity/accuracy claim.

## Arms and genuine data

- **tac_linear**: unchanged ME027 StateValueModel('tac') architecture, 135067 active trainables.
- **tac_nonlinear**: identical original native fast/slow representation and 26 real observable input features, new head as above, 135377 active trainables.
- **mlp252**: unchanged ME027 StateValueModel('mlp') with 252 hidden units and prior32, 135057 active trainables, a strong practical summary-feature baseline. This MLP does not have the exact native state/representation so it is not a causal state-off matched control. It uses the *same* event-derived 26 feature channels, and its embedding may learn different numerical values.

All use official publisher-authenticated original KuaiRec ZIP SHA256 c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c. Only already-studied original FIT uid%16==0 event timestamps before Aug01 2020 (exact source sampled 10776 true H128 histories); and already inspected original DEV Aug01–Aug14 2020 (5952 target histories). Original later-time, user1–5, MetroPT reserved and all unopened cohorts are CLOSED. The actual current offered action and known duration enter the head; the current observed watch response never enters the historical feature vector. The same three predefined seeds (14001,14002,14003), identical genuine target indices per seed, batch128, history128, same real four causal prefix targets k=32/64/96/128 (weights 1/6,1/6,1/6,1/2). AdamW lr=.001 for non-readout, readout lr=.006, decay 1e-4, grad clipping 1; exact fixed **300** steps = 38400 drawn windows / arm/seed, initial/25/50/100/150/200/250/300 reports and complete final weights/Adam/RNG. Do not train any arm longer merely to fill CPU time or overfit MLP.

**Primary preregistered DEV-only engineering criterion**, at exactly step300 (no checkpoint cherry picking): native nonlinear minus native linear Brier <0 for every seed AND mean Brier delta <=−0.0005. A separate necessary **native state-use gate**: replacing the trained nonlinear native φ(h) with FIT-first1024 sample mean φ(h) must worsen real DEV Brier >=0.0002 in every seed. This substitution preserves the FIT-average native latent logit and DOES NOT establish conditional mutual information, causal intervention benefit or OOD generalization. Calibration guard: mean ECE10 nonlinear <=mean ECE10 linear +0.003. If any condition fails the new head is rejected/unpromoted.

**Secondary**: MLP at the exact same 50 and 300 updates and all full loss curves. This is equal target presentation, parameter match within 0.25%, but **actual CPU/FLOPs differ**; no fully fair architecture superiority claim. Report measured optimizer CPU and process RSS honestly; no equal-CPU comparison in this experiment, because ME027 showed it induces divergent data exposure and MLP overtraining. The 50-step result is a registered informative fixed-budget view, never used for checkpoint selection.

## Numerical source-before-data preflight and falsifiers

- Exactly 135067, 135377,135057 model parameters and +310 head coefficients; no inert padding, no new state params.
- All three return exactly same logit prior32 before training; gradients into real native memory and A/v head become nonzero after a scripted first update.
- Verify analytic ∂ℓ/∂d and ∂²ℓ/∂d² vs autograd on constructed vectors with NONZERO head weights.
- For k=32,64,96 perturb every after-k history outcome, item/duration and target unavailable response; prediction at k cannot change. Synthetic tensors are ONLY numerical/casual-leak tests, not capacity evidence.
- Exact Brier and ECE reporting on authenticated already-exposed DEV, genuine source digest, per-seed scores, full 300-step checkpoints, model hashes and replay on all 5952 DEV probabilities per checkpoint. Halt if any hash, source split or expected parameter count mismatches.

## Promotion and next scientific test

Even if gates pass, only a **development candidate**. Never reuse opened user4/user5 or previously opened time windows as pristine. Independent evaluation requires a separately preregistered truly unopened first-exposure real cohort with cluster-robust paired scores, strong matched feature MLP/GRU/Transformer, equal params, data **and measured compute**, action costs and physical outcomes only where actually available. No commercial, agentic or Transformer-superiority claim arises from this study.

This protocol is frozen before any new model fits and must remain unchanged if ME028 fails.
