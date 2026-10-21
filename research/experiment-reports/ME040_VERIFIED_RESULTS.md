# ME040 — Verified original real-data results (no post-result retraining)

**Original frozen research run:** [GitHub Actions 39632992151](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39632992151), original preregistered source/thresholds at [research/mathematical-engineering/me040/PROTOCOL.md](../mathematical-engineering/me040/PROTOCOL.md), original trained commit e21725b63fde8a3ba320dda49e0802a2997c5956. Both jobs passed, **1,028 tests + 12 subtests**; 24 original genuine-data end-to-end fits and all 24 original saved real-development forecasts reloaded/replayed bit-exactly.

**Original training/checkpoint artifact:** [Actions artifact 12359887950](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39632992151/artifacts/12359887950), 65,680,291 bytes, SHA256 **2b11cc4c22e0ef759c09caf28ae19a7f989d14a77dc4879cfb39a21e85d99191**. Independently downloaded and matched exact published digest. Original immutable results.json SHA256 **a45d5c232fa60661033ea7b6b6b8447166e105697a428ee06a48eb5866f19d09**. ZIP contains 24 final checkpoint/optimizer/RNG states, 24 genuine historical DEV forecast matrices, all training curves and scored per-channel outcomes, source provenance, fitting and replay receipts. Original replay receipt: ME040_ALL24_ORIGINAL_REAL_CHECKPOINTS_REPLAY; numeric future labels opened in this study: 0.

## Mathematical architecture (already trained)

- Original native: twelve sequential coupled fast32/slow32/8×8 associative updates of genuine observed eight-step local/cross-channel tokens.
- Candidate paired native: six sequential **identical learned** F_TAC transitions, each on (older + 3 newer)/4, with both original input patches available at the correct causal 16-event block boundary.
- Both retain the identical full-history AR/context/horizon readout, exactly the same active learned parameters and original 128-float recurrent memory per channel. H96: 49,462 native weights vs 49,332 contextual control; H720: 130,582 native vs 130,452 contextual. All are fully trained. Unlike an execution-only kernel rewrite, this is a lossy architecture change and can change forecasts.

## Real source and evidence limits

Official publisher UCI849 Tetouan ZIP SHA256 3c4bf684161180937043a9fb65701a83d44b740b0d42f0492b6e7aec57ddbbfe and original CSV SHA256 ca8b2f30cf07dd872cb7f2aa45dac110ee38dbe5fc5910a6ff4bcc530dae3a40. All 52,416 timestamp records verified for genuine 600s cadence; **numeric values parsed ONLY in FIT+previously inspected DEV rows [0,41933)**, FIT [0,36691). The previously ME038-exposed final20% was not rescored, and CANNOT be described as new or unopened in a later experiment. Two horizons H96 and H720 (10-minute measurement steps), two registered seeds40001/40002. Four genuine FIT-only AR parents first, 500 optimizer steps each, from best validation checkpoint at100/250/500, identical parent for all downstream arms. Both arms retain genuine full eight-channel targets; power-only is last three real zone-channel MSE, NOT measured power savings, physical intervention or causal value.

## Exact historical DEV power-target results (MSE, lower better)

| Lane | Horizon | Original 12-step TAC | New 6-step TAC | Context-only |
|---|---|---:|---:|---:|
| Equal 250 FIT updates | H96 | 0.099871224 | 0.102546204 | 0.109739914 |
| Equal 250 FIT updates | H720 | 0.158654683 | 0.156542357 | 0.159900624 |
| Equal optimizer 5-second ceiling | H96 | 0.088660340 | 0.091225086 | **0.075297203** |
| Equal optimizer 5-second ceiling | H720 | 0.151020804 | 0.145417238 | **0.134053436** |

Equal-update power MSE mean over horizons/seeds: **original native 0.129262954, paired native 0.129544280, context 0.134820269**. Paired is **0.22% worse** than original but **3.91% better** than context at the same training exposure; paired H96 worsens about2.68% against original, while H720 improves about1.33%. Pair model's mean-accuracy preservation gate, including predeclared 5%-per-horizon tolerance, **PASSES**.

Equal-CPU mean MSE: **original native 0.119840572, paired native 0.118321162, context 0.104675320**. Paired improves about1.27% vs original native at fixed five-second CPU, but remains **13.04% worse** than context; mandatory fixed-CPU competitiveness gate **FAILS**. H96 favors original native over paired, while H720 favors paired over original; context is strongest at both horizons.

## Registered cost and accuracy gates

- **Speed gate PASS:** equal-update measured end-to-end optimizer training speedup (full model, not just native kernels) paired/original **1.604334× geometric**, with four matched case speedups [1.623269, 1.627434, 1.585510, 1.581671]; approximately **37.67% less optimizer CPU**. Registered speed target >=1.35×.
- **Accuracy gate PASS:** paired power-only historical DEV MSE within 2% of original native and context means, and within 5% of original native on each horizon.
- **CPU budget competitiveness gate FAIL:** paired must be at least as accurate as BOTH native and strong context control under five optimizer CPU seconds; context is substantially better.
- **Overall three-gate outcome FAIL**. **Do not promote or merge paired native or claim Transformer/SSM/architecture superiority.**

CPU-lane history exposure is NOT matched: actual H96 full native 464–474 optimizer steps; paired 773–774; context 3136–3155. H720 full native 450–451; paired 712–715; context 2347–2349. Equal-250-update lane matches true training sample streams but not physical compute. Parent training and validation should be attributed to each pipeline. The original experiment reports optimizer process CPU separately from parent, scoring and serialization, not an energy/FLOP-equivalent budget. The full model has AR/context compute and cannot claim a literal 2× wall-clock speedup from six instead of twelve recurrent transitions.

## Engineering interpretation and next build, NOT a diagnostic-only stage

This demonstrates a genuine end-to-end training improvement in throughput with broadly retained power forecasting quality. The limiting metric is now **learning efficiency per real observed example at a fixed CPU budget**, where the much cheaper context network dominates. Continue architecture-first research: build a single update-efficient TAC with a richer within-pair causal representation and significantly improved information/gradient transport, or reduce native gradient cost via mathematically valid fused kernels; require a new preregistered real-world objective and independent transfer before capability promotion. Preserve this exact failure, original selected ME038 serving exports, earlier negative evidence and the unexecuted status of any proposed improved follow-up. No component ablation or unmeasured novelty claim is necessary to build the next better model.
