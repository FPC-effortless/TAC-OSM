# ME065 — compact real-robot world-model architecture controls

**Status:** SOURCE-ONLY EXPERIMENTAL IMPLEMENTATION; no ME065 trained checkpoints, no original real-task scores, no architectural superiority. Draft on a separate branch, stacked on ME064. Created 2026-10-23 AFTER the ME064 spray-nozzle frozen holdout was revealed. **Never reuse spray nozzle as an untouched confirmation set.**

## Evidence inherited and decisive failure

ME063's native fast32/slow32/associative8x8 action-conditioned mixture (37,177 trainables) slightly edged its fully trained GRU but failed the all-baselines gate. ME064 trained that same pair on genuine original five-task FIT trajectories, froze ten original Git-owned checkpoints and scored a registered sixth task.

ME064 zero-shot original physical MAE was persistence **0.012371288**, native TAC **0.135153215**, GRU **0.148456336**, FIT servo **0.918423870**, on 80 original episodes. This shows a significant **false-movement/forecast-regime failure** against strongest persistence, NOT qualified general world modeling. TAC's episode-block averaged CI over original five-seed mean errors favors TAC over GRU, but only **2/5** individual seed pairs win; do not claim seed-robust native superiority. Source-CAL 90% whole-episode coverage at 100% with enormous widths is unqualified overconservatism. Bitwise score JSON replay initially failed due small CPU float variation; keep exact checkpoint hashes and numerical-tolerance audit separate. The distinct previously seen task and publisher remain research evidence, not real deployment or autonomous control authority.

## Concrete new controls — no replacement of native TAC

Two new **fully functional, untrained** controls share the complete ME063 physical forecast contract, horizon head, servo / persistence / learned residual expert router, output scales and 14D state-action measurements. Replacing only the observation recurrence avoids giving the controls weaker input features. The controls are:

1. **Compact Gaussian RSSM**: a 36-dimensional deterministic recurrent GRU belief and eight stochastic Gaussian latent variables. At each *observed* transition, train a posterior q(z|h,u), prior p(z|h), and recurrent GRU state update h'=GRU([u,z],h). Training adds a disclosed 0.005 KL regularizer; evaluation uses posterior means for exact repeatability. This is a *small RSSM structural control*, not a replication of DreamerV3's actor-critic, image encoder or categorical latents. No imaginary-action policy value is claimed.
2. **Input-selective diagonal SSM**: 44-dimensional stable input-dependent exp-decay recurrence, learned candidate and output gates, shared factual physical forecast decoder. This is a *selective diagonal state-space control*, not official Mamba, S4, parallel scan or a demonstrated better TAC memory.
3. **Native TAC ME063**: retains original fast, slow and 8x8 associative memory. No conversion into an SSM.
4. **Fully trained matched GRU ME063**: established nearby recurrence baseline.

The shared architecture's names are technical taxonomy rather than claims of capability. The RSSM's architecture-specific KL must be disclosed when interpreting any eventual performance contrast. A same-physical-objective RSSM or KL-strength sensitivity may be needed to attribute a memory mechanism.

**Additional TAC contender:** ME056 Concept-Swarm-CED (six persistent 15D concepts and four shared-weight decoder refinements) is retained as a separate candidate, not quietly dropped. Its original parity preflight on real robot FIT failed by **61.32% profiled FLOPs** and it has no qualifying trained comparison. Only admit it to this common benchmark after a genuine FIT source-only matched preflight with active parameters and operators within the same compulsory gate. Do not claim it is one of the best *trained* variants.

## Source, frozen resources and first experiment

- Source: identical genuine 160 original FIT episodes across the five previously exposed ME064 robot tasks; exact SHA-256 **4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776**. No raw simulated or generated trajectories as training/success evidence.
- Model API: identical 14D physical current state, 14D factual action, observed authority, 64 preceding genuine physical transitions and 16 real observed future transitions, original FIT-only normalization and servo identified parameters.
- Frozen default geometries **before** profiling: native TAC (32,64), GRU (44,82), compact RSSM (44,82) with 8 stochastic dimensions, selective diagonal SSM (44,103). They are not certified as parity-matched merely because expected parameter counts are close.
- Stage 1: independent **original FIT only** preflight computes real active parameter counts, reported PyTorch-supported forward/backward FLOPs, and parameter gradient coverage for every arm using precisely the same original sample. If any pair differs by **1% or more**, or active gradients are missing, **no comparative training** may start. Record all failed FIT-only results in Git. PyTorch profiler does NOT count complete all-kernel training/optimizer hardware FLOPs.
- Stage 2: once prospective experiment registration and source-only parity pass, preserve identical FIT sample schedules, **600 actual optimizer updates per seed**, batch size 8, five paired seeds **65001–65005**, optimizer hyperparameters and selection protocol; report measured CPU efficiency but **CPU times need not match**.
- Stage 3: preserve **all 20 original 4-arm x 5-seed** full trained weight/optimizer/RNG checkpoint bytes in TAC-OSM Git **before opening any independent source CAL or new target label**. Actual original source publisher hashes, artifact code, dependencies, member manifest, timestamps, splitting, sample order and scientific receipt must be recoverable. A clean independent Git checkout must restore/recalculate weights and score; local tarballs, GitHub Actions artifacts, hash-only promises or synthetic checkpoints are not substitutes.
- Stage 4: a new task chosen from publisher metadata among **previously untested** `plug_charger_into_power_strip` and `replace_tissue_box_and_pull_out_first_tissue` candidates *before opening their outcome trajectories*. The selection must be deterministic, documented, metadata-only and irrevocably frozen, with any ineligible candidate rejected before exposure. A second previously untouched task remains reserved. The exposed `install_spray_nozzle_onto_bottle` cannot be used as a confirmation task or for hyperparameter/model selection.
- Stage 5: score real physical MAE plus per-horizon, motion-regime, false-positive change, calibrated sharpness/coverage and factual authority/takeover subsets. Always compare persistence, FIT servo, the strongest matched neural control, and each other eligible candidate. Episode-block cluster inference AND seed-pair uncertainty must be reported; an episode-block-only positive result with low seed wins is not enough.

**Primary scientific gate:** a TAC variant must improve >=2% over *every* matched learned comparator and the strongest applicable untrained physical reference on a previously untouched whole authentic robot task, with the paired recording-block 95% CI upper limit below zero and robust independent-seed consistency. In addition require useful source-calibrated interval sharpness/coverage, *not* wide but trivially covering bands. Secondary scores cannot override a failed primary gate. If a comparator's compute/parameter matching fails, label results *engineering diagnostics only*, not controlled architecture superiority.

## Scientific restrictions

- Original logged robot action prefixes are **factual observational**. They do not prove that unexecuted alternative controls work or a learned planner has higher intervention value. Only actual authorized prospective action/outcome studies can establish that.
- Posthoc ME064 motion grouping may diagnose failure, but no threshold learned from the spray-nozzle labels is eligible for validation on those same episodes.
- Capability before micro-efficiency. Equal parameters, source/data, training FLOPs budget and access to all original checkpoint bytes are compulsory. CPU runtime inequality is disclosed rather than disqualifying.
- Woyengi remains fail-closed advisory-only; Veritas simulations can diagnose but cannot substitute for authentic real decision effects.
- Do not confuse a new experimental architecture with a successful improvement. ME065 preflight is an untrained numerical and resource experiment.

## Established foundations and provenance

DreamerV3 uses a deterministic plus stochastic RSSM with imagined outcomes and policy learning, far beyond the present micro RSSM control: https://doi.org/10.1038/s41586-025-08744-2

Mamba establishes input-selective state-space update mathematics; this control intentionally implements only a small stable diagonal recurrence, **not** full Mamba: https://arxiv.org/abs/2312.00752

Both PSRs and uncertainty-filter methods remain external conceptual/practical baselines; a real predictive-state control requires measurable future-test state and legitimate action-observation history, not relabeling an ordinary hidden vector: https://arxiv.org/abs/1501.06528

Original comparator source: `src/tac_osm/me065_micro_world_controls.py`.
Source-only preflight: `scripts/me065_prefit_original_fit.py`.
Mandatory original research rules: `research/REAL-WORLD-EVIDENCE-RULES.md`, `research/REPOSITORY-REPLAY-AND-CHECKPOINT-RULE.md`.
