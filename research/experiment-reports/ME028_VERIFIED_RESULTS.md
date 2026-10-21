# ME028 original verified results — nonlinear native prediction head works, learned-state guard FAILS

**Source/runner:** original GitHub Actions [39530605976](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39530605976), prefit/predefined experiment commit [275b09d](https://github.com/FPC-effortless/TAC-OSM/commit/275b09dceee4a84eb4ad508039adb55c5306b762). The original numeric-only test was corrected *before* FIT, because the first falsifier incorrectly changed the **known future offered item/duration** rather than its unavailable outcome. The corrected nonzero-head causal falsifier and 917 regression tests PASS. No fitted weights changed after training or gate criteria retuned.

**Original retrievable raw evidence:** [GitHub Actions artifact 12315329303](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39530605976/artifacts/12315329303) — nine trained model weights/optimizer/RNG checkpoints, all nine final DEV prediction arrays, full trace in results.json, fit log and independent replay log. Original ZIP SHA-256 **b0025dfea2a6f12f8d4aa28ca2dd80cb26e777092f9d9f69baf555da678e08b2** (verified locally), inner original results.json SHA-256 **ccab7673776088f2541a16ed866a54de615bc2cc50809c740db60fa6d5ce0c91**. First downloaded artifact independently SHA-matches GitHub's published artifact digest.

Original FIT/DEV source: authenticated publisher KuaiRec ZIP SHA-256 c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c; 10,776 FIT and 5,952 **previously inspected** DEV real H128 windows, user%16=0. No unseen user-4/5 labels, newly reserved test, or causal intervention outcome accessed. All models use 300 genuine target-window index-aligned optimizer updates, batch128, four actual causal prefix-outcome targets. All model params within 0.24%; **FLOP/CPU match fails**, don't claim equal-compute architecture superiority.

## Fixed-300 step historical DEV scores (Brier lower better)

| Arm | Active trainables | Seed 14001 Brier | Seed 14002 Brier | Seed 14003 Brier | Mean Brier | Mean ECE10 | Mean optimizer CPU sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unchanged native TAC linear | 135,067 | 0.168340878 | 0.165751732 | 0.165124552 | 0.166405721 | 0.02855308 | 25.8749 |
| Native TAC + 12-unit nonlinear observable head | 135,377 | **0.166568092** | **0.163373150** | **0.162653575** | **0.164198272** | **0.02765930** | 26.1945 |
| Summary-feature MLP252 | 135,057 | 0.165563666 | 0.167248851 | 0.163344183 | 0.165385567 | 0.03973927 | 2.1201 |

**Primary paired nonlinear minus linear real historical DEV Brier:** seed14001 **−0.001772786**, seed14002 **−0.002378582**, seed14003 **−0.002470977**; mean **−0.002207449**, i.e. ~1.33% relative improvement compared to 0.166406 mean baseline. The registered accuracy gate **PASSES**: all seeds improve and mean delta is better than −.0005. **Calibration guard PASSES**: mean ECE10 improves by −0.000893779 (new 0.027659 vs original .028553). The nonlinear TAC surpasses the 252-unit MLP at step300 on mean dev Brier by 0.001187295, **but this is NOT a fair equal-compute architectural victory**: the MLP uses dramatically less CPU, and dev has been used repeatedly in the program.

**The mandatory state-value gate FAILS:** replacing frozen learned phi(h) at time 128 with the FIT-first1024 mean phi(h), while keeping all true causal observables/prior constant, changes DEV Brier by:
- seed14001: **−0.001040704** (replacing actual state IMPROVES forecast!)
- seed14002: +0.000459793 (state helps under this diagnostic)
- seed14003: +0.000268087 (state helps under this diagnostic).

Registered threshold >=+0.0002 for every seed; seed14001 decisively fails. Thus **ALL-GATES PASS = FALSE**. The new head is **NOT PROMOTED** even though aggregate forecast accuracy and calibration improve. These state-feature interventions have OOD caveats and don't estimate information-theoretic conditional state value. Retain ME024 RMS-only as the previously established reference; retain ME028 as unpromoted possible future candidate for genuinely independent evaluation after a scientific plan.

## Dense registered, unselected learning curves

| Step | Native TAC linear mean Brier | Native TAC nonlinear mean Brier | Feature-MLP252 mean Brier |
|---:|---:|---:|---:|
| 0 | 0.20242651 | 0.20242651 | 0.20242651 |
| 25 | 0.19480418 | 0.18694658 | 0.16478497 |
| 50 | 0.18610281 | 0.16655109 | 0.16450833 |
| 150 | 0.17249014 | **0.16365411** | 0.16446038 |
| 300 (registered primary) | 0.16640572 | **0.16419827** | 0.16538557 |

At 50 updates the MLP still beats nonlinear TAC, even though readout improvement markedly accelerates TAC versus its own earlier linear-head version. At 150, native nonlinear is very strong; **do not cherry-pick 150** because selection rule fixed 300 and historical DEV is not independent. Nonmonotonicity is preserved, not retuned away. Equal CPU is NOT a separately run primary because the ME027 CPU match induced extra MLP overfitting.

## Verification and interpretation boundaries

- CI [39530605976](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39530605976) two jobs **success**. **917 repo tests passed** and full 64-state, nonlinear readout/derivative, TorchScript and nonzero-head causality checks PASS.
- All nine final model checkpoint SHA256 and all final prediction-array SHA256 verified before replay; all **53,568** authentic historical DEV forecasts replay on genuine publisher data; original Brier reconstructed from replay within tolerance.
- No model weights changed, no new test cohort or experimental arm added after original training. Raw artifacts preserve original failed and successful controls.
- The work is motivated by Netflix GenRec's task-aligned prediction and by Sep-Latent / NCM's emphasis on useful compressed predictions, but it is specifically a **readout-only** test; no Transformer-like text, latent step tokens, hierarchical coding, agentic control or SSM model was implemented.
- Differences in CPU, learned representation and nonlinear readout prevent interpreting these scores as proof the TAC memory has more information than simple features, is superior to a Transformer, or is commercially ready.

**Next falsifiable direction, no fit yet:** isolate native state contribution without an OOD mean swap by (a) jointly training an identical nonlinear observable head with and without recurrent-state access while keeping treatment-active parameter accounting clear, (b) conducting a frozen-head conditional residual probe with user-disjoint genuine FIT histories, (c) ensuring a truly unexposed real future cohort is preregistered before training/model selection, and (d) only then testing event-boundary latent compression or adaptive silent steps if operational verified decision outcomes exist. Do not repeat known positive/negative experiments or manufacture synthetic capability evidence.
