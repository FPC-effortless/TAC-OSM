# ME-015G — Compulsory strongest conventional GRU control, same task, fresh real user cohort

**Registered 2026-10-20 BEFORE opening ANY user IDs satisfying `user_id % 16 == 2`** in this TAC-OSM research. Both earlier ME014BF and ME015F heldout cohorts are now opened (respectively uid%16==0, uid%16==1) and must NOT be reused as fresh results. This test keeps KuaiRec big_matrix, full-watch target, 32 prior real interactions, current video action/duration, identical chronological future period Aug14–Sep06, original trained weights and source checksum unchanged. No further fits permitted.

**New genuine never-trained user cohort:** `uid%16==2` at factual logged action times 2020-08-14 inclusive to 2020-09-06 exclusive; causal prior32 responses available from the same user strictly earlier than current response. No user ID in neural input. This cohort's labels have not been parsed by any original ME014, ME015 fit or ME015F uid1 evaluation code (which explicitly excludes all residues except 0 and 1).

**Frozen original checkpoints compared in 3 matching seeds (all EXACTLY 135050 trainable parameters)**:
1. Native TAC original from ME015 original run 39398038257;
2. Native TAC with BOTH chrono bias and factual prior logit from ME015 run 39398038257;
3. GRU + identical causal prior logit from ME015 run 39398038257;
4. **Original strong, UNANCHORED GRU** from original ME014B run 39394500641, exactly same params, FIT records, seeds and 300 AdamW steps.
The original GRU must be scored rather than ignored because anchoring unexpectedly harmed it on old DEV. Verify ALL original archive/source SHA256 digests and all candidate checkpoint weights **before any heldout user ID %16==2 record** is read. Do NOT retrain or select the best seed from results. Include factual prior32, training-only item and global rate baselines. Score Brier and logloss across every real test exposure, 3 paired seeds; report history lesion and CPU training/inference cost. No artificial synthetic success.

**Go/no-go qualification:** Engineered TAC must beat **both anchored and original unanchored GRU** in all 3 paired seeds and beat causal user-rate baseline, otherwise reject the claim it outperforms GRU in this task. Even passing qualifies only real logged *factual* prediction across disjoint user populations, NOT training-FLOP fair win, unseen action counterfactual, decision-cost utility, System 2 or commercial autonomy. No model/refit tweak after first label exposure and no auto-merge. TAC identity remains a native nonlinear two-timescale recurrent world-state model, not an SSM.
