# S1MOTION006 — causal measured-velocity motion-head feature ablation
**2026-10-26 prospective preregistration; NO real fit, DEV, TEST, result, or promotion yet.**

## Causal observational intervention
The actual previous measured state x[t-1] is ALREADY available from trusted source at time t. A source-agnostic 14D auxiliary evidence substitution needs **zero extra learned parameters and zero extra carried FP32 state**, preserving native 95,907 active weights and 664-byte stream state. Reuse current model's 198 -> 12 -> 15 auxiliary head unchanged:
- CONTROL: auxiliary features [fast88, current derived slow88, old associative read8, learned core one-step state estimate14], the exact original S1MOTION005 wiring
- TREATMENT: same features except final14 = learned core one-step estimate14 + (actual measured x[t] - known x[t-1])14.
This preserves absolute learned state information while adding a direct permitted one-step motion path; only auxiliary head input changes, no future outcome, next measured state or proposed action. Point forecast is **exactly the same at identical weights**; training may later differ because auxiliary gradients affect shared memory even with frozen projection budget. All other parameters, sample windows, event loss and optimizer remain unchanged.

Testing a direct velocity feature does NOT prove the event label's next velocity can be causally predicted, and real logged actions remain evidence from earlier execution, not do(a). The classification label is a FIT-derived high-next-movement proxy, not real safety or successful manipulation.

## Genuine prospective external-source plan
Use **NEW** LeRobot `lerobot/aloha_static_coffee` publisher task (MIT per current card; reported 50 real episodes, 55,000 frames, 50 Hz, 14-joint bimanual state and actions) subject to **first immutable publisher revision and 25 individual FIT file SHA256 audit** before any optimizer update. Live HEAD layout MUST be verified; fail-closed rather than auto-repairing revision assumptions. No source episode data outside FIT 0..24 available to training/preprocessing; DEV25..34 only after all original model/optimizer/RNG/checkpoint SHA frozen; TEST35..49 never downloaded. Verify actual 1100 frame continuous clock, channel dimensions, episode IDs, source allowlist and past-action alignment. No camera/video or next-frame label reaches forward model.

Set seeds (36001,36002,36003), equal within-TAC 192 AdamW update steps, batch8 × window32, matched source windows and initial model weights, identical projected `rho=.10, beta_cap=.25` two-grad algorithm, FIT-only normalizers/event label threshold, no adaptivity based on held DEV; each control and treatment original trained independently. Profile full operation training FLOPs, AD calls, source counts and CPU medians, compare parameter, original checkpoint and actual prospective budget. Differences in extra vector add may prevent STRICT exact training FLOPs; label any mismatch and no fully equal-compute inference when controls differ. Optionally run matched dummy ADD arithmetic in both arms if required, but must record actual profiler and optimizer backward compute.

**Acceptance gate, prereg before DEV:**
1. Every seed: motion-proxy AUPRC(treated) >= 1.05 × AUPRC(control) AND >= source label prevalence; report per episode paired resampling and Brier/AUROC, not just AUPRC.
2. Every seed: source physical next-state MSE(treated) <= 1.01 × MSE(control), and <= 0.99 × causal velocity MSE; high-motion MSE(treated) <= 1.05 × velocity on all seeds.
3. No meaningful observed future-target gradient leaf in forward; score source future raw-row poison must leave decisions invariant. No TEST access until protocol amended on independent source, not silently.
4. If full FLOP/parameter/data match gate fails, **do not report a qualified fair architectural superiority** even if descriptive outcomes improve. 
The old S1MOTION005 screw-driver DEV was opened and can NEVER become an unseen prospective test for this new feature. Preserve its failed 2-of-3 full gate and all negative artifacts.

## Separate persistent memory research
No combined velocity+slow model to claim an unidentifiable improvement. S1TEMP006/007 will test a separate **fixed 88D fast bank reallocation, 72 additive +16 persistent causal EMA**, 64D associative and 14D previous actual observation: keeps 166 floats and 95,907 weights but changes both streaming and parallel scan. Equal parameter/FLOP/data/seed gates and real >512-frame temporal measures required, including actual memory ablations and a matched fast-only control; engineering checks come BEFORE fit. A negative result is a result.

**Stage status:** zero-parameter source code/tests exist; no evidence any event ranking improved yet.
