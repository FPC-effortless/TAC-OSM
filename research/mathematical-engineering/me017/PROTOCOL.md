# ME-017 — Optimization stability of the SAME native TAC against strongest GRU

**Preregistered October 20, 2026, before ANY ME017 training and before any user_id modulo16=4 labels are read.** No new recurrence feature, memory expert, entity slot, SSM replacement, new task or dataset: the ME015 chrono+causal native recurrence was validated on untouched authentic user1/user2/user3; ME016 fixed fast-to-slow identity modification was rejected on authentic historical development and its first user3 cohort. The remaining hypotheses are learning stability, gradient credit, and non-monotonic generalization from 300->600 steps.

## Single mathematical hypothesis H1 — averaging stochastic optimization noise

For every learned parameter tensor, save the full 600-update AdamW trajectory final iterate `theta_600` and update an independent exponential moving average of the SAME model weights:

`theta_EMA,0=theta_0;quad theta_EMA,t = 0.99 theta_EMA,t-1 + 0.01 theta_t`.

The value beta=0.99 is fixed BEFORE any ME017 results; no tuning from already opened users. In a hypothetical independent stationary weight-noise process, EMA variance is reduced approximately by `(1-beta)/(1+beta)`; this idealized approximation is NOT a guarantee for real nonlinear nonstationary neural optimization or a true world-state memory proof. The sigmoid/tanh convex recurrent-state box invariant remains true for the averaged weight model too; empirical predictive correctness/calibration is not mathematically guaranteed.

Save both final and EMA weights for each seed. No new model trainables (EXACTLY 135050) and no extra optimizer gradient update: EMA uses only fixed linear parameter arithmetic after each regular update. It creates a second inference checkpoint per arm but not an ensemble of two simultaneously deployed models.

## Three original equal-parameter, equal-training-data arms

(A) existing native TAC with ME015 chrono gates (fast logit1/8, slow logit1/24) AND causal prior32 user response base-logit;
(B) **original unanchored exact-sized GRU** (stronger classic GRU baseline);
(C) **same prior-logit anchored GRU**, identical historical information.

Keep everything else original from ME016: KuaiRec publisher ZIP MD5 261550d472c48eff4990fb13c0e5bcf7 and SHA256 c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c. Original FIT user_id modulo16=0 before 2020-08-01, 68,825 genuine logged action/history target windows; old Aug01-Aug14 DEV already exposed (106245 windows), no future Aug14-Sep06 source responses from original or user cohorts1/2/3. History has 32 genuinely preceding offered-video actions and observed outcomes, current video and known duration but NEVER current target. All model trained active parameters **135,050 exactly**, three fixed paired seeds 14001,14002,14003, **600** AdamW updates, batch128, lr.001, decay .0001, global clip1, same random sampled real FIT batch IDs in every seed; 9 independent original fits and **18** full-model SHA256-verified final/EMA checkpoints. Measure CPU seconds, EMA update overhead, per-iteration objective and true history gate statistics. Matching parameter counts / training batches / steps does not establish identical CPU/FLOP budgets. Report measured dissimilarities and do not call it a strictly fair equal-FLOP head-to-head.

Historical Aug01-Aug14 DEV used only for diagnosing whether EMA changes *already exposed* generalization and for software/source/shape audits; never for selecting beta or best checkpoint.

## Single future first-exposure real-world falsification

Reserve **user_id modulo16 = 4** on the **same genuine KuaiRec original action-response task**, future 2020-08-14 to 2020-09-06 exclusive (pre-register exact source, no label filtering by outcome). This disjoint set has not been opened by prior ME014–ME016 source parsers, restricted to users modulo16 in {0,1,2,3}. The future scoring must be in a **separate workflow/PR after original model hashes freeze**, verifying all 18 hashes and matching official source ZIP digest BEFORE the user4 CSV rows are parsed, with zero fine-tuning and the same 32 causal prior observations.

Evaluate all three seeds of all three arms with BOTH their final and EMA checkpoints, plus true causal prior32 user watch-rate, original TRAIN-only video rate and global rate. Primary factual Brier, secondary Bernoulli log-loss/calibration, history removal sensitivity and per-user averages. User-cluster resampling if storing per-user log-loss aggregates without storing identifying records. **Do not score prior already opened user1/user2/user3 cohorts as fresh**.

Strict EMA-TAC gate: averaged native TAC must beat its own final checkpoint in **3/3 seeds**, must beat each original and EMA GRU in **3/3 paired seeds**, and must beat causal user-rate baseline. Otherwise reject EMA as robust improvement even if mean is favorable. In an additional compute-matched GRU run, allow GRU extra optimizer steps up to native original wall budget and report extra data-exposure unfairness; do not perform that yet or claim matched-compute success from this study.

No independent signed action receipts, propensities, unshown-video counterfactuals, commercial utility, System2, world-model generality or autonomous CASM safety can be inferred from observational full-watch prediction. Preserve EVERY original research idea and negative result, including rejected ME016 identity pathway. Draft, no automatic merge, TAC not converted to an SSM.
