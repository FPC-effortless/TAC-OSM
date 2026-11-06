# TAC-OSM integrated world-state research — full authentic robot training source

Registered October 22, 2026. Source-only qualification built on the already merged native `PredictiveStateAgent`, publisher episode audit and frozen 325 FIT / 80 DEV / 100 TEST blocks. This is **not** another architecture micro-variation, a synthetic benchmark, or training. The project question remains whether a genuinely trained, persistent and verification-correcting TAC world-state model can improve real future physical-state prediction and downstream externally verified action decisions over a fully **parameter/data/compute-matched** established recurrent world model.

## Reuse existing research; no rediscovery

Known foundations: action-conditioned world-model training (DreamerV3, TD-MPC2), recurrent replay and burn-in (R2D2), human-control data and correction imitation (HG-DAgger), and mechanical recurrent-state engineering (Mamba-2). None is claimed novel here. This phase QUALIFIES all source FIT frames and records actual supervised-label support; it does not trial new small memory gates or reconstruction losses.

- [R2D2 recurrent experience replay](https://willdabney.com/publication/r2d2/)
- [DreamerV3](https://doi.org/10.1038/s41586-025-08744-2)
- [TD-MPC2](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html)
- [HG-DAgger](https://arxiv.org/abs/1710.02840)
- [Mamba-2 state-space duality](https://proceedings.mlr.press/v235/dao24a.html)

## Original source and honest evidence boundaries

The *real human intervention robot trajectories* are from [BodenAI RW-RL-HIL-Dataset](https://huggingface.co/datasets/BodenAI/RW-RL-HIL-Dataset), immutable revision `9290f5a96dc91ea58256e81d9ce2dd9621890b51`, original task `scoop_cat_litter_into_trash_can`. Original 505 episode source metadata already qualified, and 20 FIT episodes already inspected in prior merged PR322. This next step independently audits **every one of the 325 original training episode parquet files**, not a randomly selected subset, and never opens frame-level DEV (80) or TEST (100) files.

For each authentic FIT physical trajectory it verifies original publisher episode index, exact frame count, strictly increasing source timestamps, all physical 14D observation and logged action values, human-policy control authority, and original metadata intervention counts. It SHA256-pins every original FIT episode, retains aggregate original frame counts, and records counts of real factual next-state and human takeover-onset targets at 16, 64 and 256 **truly observed history frames**. The onset label is available only when *policy* controlled the robot at t; human-controlled frames are masked as unlabeled for onset rather than forced to zero. The 325/80/100 cohort partition is unchanged and based only on IDs, not outcome values.

No raw robot frame, patient/user identity, video or copyrighted source file is committed or uploaded. Only source hash receipts and aggregate statistics are archived, with full original data maintained on ephemeral private GitHub runners. The fixed original metadata file itself contains published per-episode intervention summaries, so metadata-access constraints must not be misrepresented as preventing all knowledge of test-label aggregates. The actual future episode **frame-level** labels are not fetched. No operator takeover is treated as independent proof of physical failure or reward; no human skill/success/cost labels are fabricated.

## Actual next major experiment — one integrated capability study, not micro-ablations

1. **Joint trained objective**: learn from actual FIT-only `(x_t, a_t, x_(t+1), policy/human authority_t)` episodes; persistent TAC fast/slow/associative belief updated from observed physical frames, supervised factual next 14D state and takeover-onset only while human not already in control, plus *postprediction* correction using measured physical discrepancy when it truly arrives. No future action in a prediction input.
2. **Established recurrent world-model comparator**: same 14D physical state/action channels, same expected before-action information, same feedback timing, identical 325 FIT episodes and sample order. No scripted planner or fixed access to a privileged state. Match trainable and deployed parameter counts within 1% and tune meaningful model dimensions fairly; no dummy unused parameters.
3. **Pretraining measured resource profiler**: run on the same actual train-only source batches with no optimizer updates or source-label fitting; report prefit compute budgets and actual FLOP estimates, CPU efficiency, peak memory and equal preprocessing/data exposure. CPU time differences do not disqualify matched budgets; changing model widths or compute allocation requires newly frozen manifest *before training* and the same privileges for every rival.
4. **Data and exposure**: same 5 seeds, original train split membership, strictly original states/actions, 256 authentic history frames, identical optimizer-step count, example presentation order, tokens, tuning trials and validation opportunities. Test episode frame-level records remain unavailable until all model weights/optimizers/checkpoint SHA receipts freeze.
5. **Evidence gates**: independently held-out 14D next physical-state MAE and RMSE, calibration/error by policy or human authority, actual onset PR-AUC and class base rate where positive cases exist, and predictive improvement after verified corrections at 16/64/256 true horizons. Report matched-resource head-to-head only if the source-controlled `FAIR_COMPARISON.json` passes **`scripts/validate_research_parity.py --phase preflight` and `--phase final`** on real receipts. Plain constant-state reference is allowed only as an explicitly unmatched practical baseline.

This particular data does **not** contain verified real return, task success, task completion, or intervention assignment propensity. Therefore even a positive trained next-state forecasting result cannot qualify an autonomous, causal reward-optimal controller. That is a separately blocked decision-value lane that requires new external or prospectively collected genuine evidence.

## This stage's acceptance, failures and nonclaims

The original publisher-source qualification passes only after all 325 FIT parquet files are physically downloaded and audited, their original file hashes are retained, history H256 factual next-state support is demonstrated, actual original human control/onset counts are recorded, and DEV/TEST trajectory files were never opened. If any FIT source fails, the entire stage FAILS and no training follows. This audit does not open the real TEST labels, train even one model, perform optimizer steps, calculate heldout prediction metrics, or pass the matched parameter/data/training-budget comparator gate.

**Research novelty is measured by integrated capability, not the existence of an action-conditioned recurrent neural model.** The next study must be a full trained TAC-versus-matched-established-agent experiment, not a sequence of tiny memory tweaks.


> **User-specified scientific correction (2026-10-22):** Matching CPU execution
> time is **not** compulsory. Match trainable parameters, genuine dataset and
> exposures, training/tuning budgets and evaluation conditions. Record CPU time
> as an efficiency finding, not a fairness PASS/FAIL gate. The standing
> `research/REAL-WORLD-EVIDENCE-RULES.md` amendment controls future runs.
