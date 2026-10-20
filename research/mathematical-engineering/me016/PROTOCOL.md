# ME-016 — Repair slow-memory credit assignment in native TAC, not add components

**Frozen research protocol: 2026-10-20, BEFORE opening any user_id %16==3 response labels.**

## Why this is the next mathematical engineering target
ME014B: real full-watch test, original native TAC Brier ~0.196248, strongest GRU ~0.194484. ME015 introduced ONLY gate-timescale initialization (fast logit1/8, slow logit1/24) and a deterministic causal prior32 watch-response logit. Two genuine disjoint user cohorts show the combined native improves over original native and anchored GRU. On user2 (40,396 real examples), combined native TAC mean Brier 0.19234283 vs original unanchored exact-parameter GRU 0.19362136; however TAC loses original GRU seed14002 0.19255521 vs 0.19190594. User-mandated all-three seeds superiority **FAILED**. No claim of conclusive superiority or equal compute.

Inspect prior original real checkpoints: actual slow gate write probability ~0.043 with corrected chrono gates; earlier ~0.020. The slow memory has to learn `W_s[s_t,z_t]` before action loss can differentiate useful fast innovations. This hypothesizes **weak temporal credit assignment into the slow state** (not proven by output norms). Test direct gradient throughput and improve it without increasing trainable parameter count.

## Single architectural hypothesis: identity-coupled fast-to-slow innovation
Existing native TAC:
`q_t=tanh(W_xx_t+W_zz_{t-1})`;
`z_t=(1-g^f_t)z_{t-1}+g^f_t q_t`;
`s_t=(1-g^s_t)s_{t-1}+g^s_t tanh(W_s[z_t,s_{t-1}])`.

**Proposed TAC-I native state update** (no new weights, no GRU/SSM conversion):
`s_t=(1-g^s_t)s_{t-1}+g^s_t tanh(W_s[z_t,s_{t-1}] + z_t)`.
The added `+z_t` is an unlearned **identity innovation path**, dimensions fixed 32 fast and 32 slow, with fixed coefficient 1.0 registered a priori. This directly changes slow-state derivative wrt current fast state to include `g_s diag(1-tanh²(...)) I`, instead of relying entirely on `W_{s,z}`. **No gradient-lower-bound or long-memory theorem** follows because tanh can saturate, gates can close and other terms can cancel. Invariant `z_t,s_t ∈[-1,1]^{32}` remains exact by sigmoid convex interpolation and tanh bounds.

No new learnable parameters. Exactly 135,050 parameters for all arms (same current video embedding, current action and 32 previous factual observations); same causal smoothed prior32 logit and original two-timescale gate initialization for the two native arms.

## Source, registered fair controls and budget
Original official KuaiRec ZIP pinned MD5 `261550d472c48eff4990fb13c0e5bcf7` and SHA256 `c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c`. Parse only source rows for user_id%16==0 and timestamp <2020-08-14 for training/historical development. FIT <2020-08-01, historic developer diagnostics Aug01–Aug14. No prior user0 future or already opened user1/user2 reactions read. User ID is an external grouping field only; neither model sees it.

Original **four same-task architectures**, exactly 135050 trainable weights each:
(A) ME015 combined chrono+prior native (unchanged);
(B) new identity-coupled native TAC-I (only mathematical slow proposal changed);
(C) original anchored exact-sized GRU;
(D) original unanchored exact-sized GRU (same active 68-parameter video-action refinement).

Seeds 14001/14002/14003. Each arm exactly **600** AdamW optimizer updates, batch128, BCE-with-logits, optimizer lr0.001/decay0.0001/clip1, same deterministic example indices (seed+300), all FIT windows and labels identical. 600 steps explicitly probes convergence beyond the original 300-step undertraining limitation. **Equal optimizer-step/data/parameter budgets do not establish equal FLOPs or wall-clock time**: record total training CPU time, median batch time, and call FLOP-equal comparison unqualified unless measured. Save original 12 checkpoints, train batch loss at steps 1/150/300/450/600, genuine history gradient telemetry at steps0 and600 (slow pathway gradient vs fast). For the latter choose 128 real FIT examples and measure `d(loss)/d(z_t)` at historical steps t=0,8,16,24 (not data-generated) via retained gradients. Never use hidden test reactions to select hyperparameters.

Control diagnostic: separate within-architecture A vs B, same parameters, optimizer steps and data; a consistent improvement in 3/3 paired seeds means the identity mathematical pathway helps historical development. Baseline C,D prevent beating an artificially weakened GRU.

## Future valid first-exposure evidence
**Entirely disjoint real user cohort** user_id%16==3 and factual offered video reactions dated 2020-08-14 UTC through 2020-09-06 UTC exclusive. This cohort has not been opened by previous scripts limited to user residues 0,1,2. A separate later PR may score *frozen* 12 original model weights and original fitted train-only global/item + causal prior32 user-rate comparators after all hashes are checked. Zero test optimizer steps, all three paired seeds. **Gate:** candidate B beats A and BOTH GRUs in all three seeds *and* causal user-rate baseline, with equal active parameters and matched optimizer updates, followed by equal compute qualification. A result with any failed comparison must be recorded as failure; do not tune using opened heldout user3 outcomes.

Model remains observational factual action-conditional probability under original exposure policy, NOT verified counterfactual optimal recommendation/repair, autonomous CASM action payoff, independent receipt verification, or System2 cognition. All retained unified plan/ME001–ME015 concepts and positive/negative results preserved; research draft unmerged.
