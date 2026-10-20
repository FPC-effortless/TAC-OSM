# ME-016 — Original 600-step authentic KuaiRec native memory engineering outcomes

**Scientific conclusion (historically opened development data):** the proposed parameter-free fast-to-slow innovation **FAILS** to improve the existing native chrono+causal TAC. Its 3-seed mean Brier is slightly higher and its per-seed paired win is only 1/3. The existing native TAC is strongest on mean among both conventional GRU controls on this already explored dev period, but this is **NOT** independent heldout superiority and **NOT** complete equal-compute control.

Original execution: successful [GitHub Actions 39412071932](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39412071932), original source SHA `e12982eb59a2b820855888b2df0f74d2553abb40`; original artifact `12259967443` (`me016-original-real-innovation-tac-strong-gru`). The first attempted preflight run `39411925263` failed **before training** due solely to stale `assert len(ARMS)==5` after reducing to four registered arms; corrected to four with no mathematical protocol change. All **12 PyTorch checkpoints** from the successful original fit have matching recorded SHA256 digests independently verified from the archived ZIP.

Official genuine KuaiRec original source publisher ZIP MD5 `261550d472c48eff4990fb13c0e5bcf7`, SHA256 `c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c`. Original FIT users `uid%16==0`, 68,825 original causal action-response history windows; previously inspected DEV 106,245 windows. All models EXACTLY **135050 active trainable parameters**, same three fixed random seeds 14001/14002/14003, 600 same real FIT batches per seed, batch128, Bernoulli BCE loss, AdamW LR.001, decay.0001, gradient clipping1. The untrained user1, user2 and new user3 future test reactions were **NOT parsed** in original model fitting.

| 600-step trained arm, historical DEV | seed14001 Brier | 14002 Brier | 14003 Brier | 3-seed mean | mean training CPU seconds |
|---|---:|---:|---:|---:|---:|
| **Existing chrono+causal native TAC** | 0.20095393 | **0.20078953** | 0.20055109 | **0.20076485** | 10.74 |
| New identity-coupled slow native TAC | 0.20106740 | 0.20082521 | **0.20049872** | 0.20079711 | 10.84 |
| Anchored exact-sized GRU | 0.20284024 | 0.20184538 | 0.20214053 | 0.20227538 | 9.92 |
| Original unanchored exact-sized GRU | 0.20260482 | 0.20129580 | 0.20088767 | 0.20159610 | 9.86 |

All values are mean Bernoulli Brier on **historical DEV**, not independent test. The **H1** candidate native slow identity innovation loses to unchanged native in seed 14001, 14002 and wins only 14003. Since one proposal failed the targeted improvement gate, **do not promote it as a new default TAC recurrence** even if later first-exposure results were selectively favorable. The original anchored native retains practical advantage vs GRU on this dev split, and its trained gates remain conservative (mean slow update 0.0398 after longer 600-step training).

The structural mathematical facts are independent of performance: with z and s initially in [-1,1], each candidate update `s'=(1-g)s+g*tanh(W[s,z]+z)` remains bounded, and the extra z path supplies a direct derivative term within tanh. A nonzero derivative path is NOT proof of gradient lower bounds, long-memory retention or better generalization. The preflight confirms active parameter count and finite differentiation; dedicated exact real-gradient-tap measurements were NOT completed despite being in the preregistered diagnostic aspirations, and must not be presented as measured evidence.

The preregistered unseen user-id remainder3 first future outcome test is separate draft PR, using the immutable original weights and source hashes. Do NOT evaluate the already-opened remainder1/2 as fresh evidence or tune against these results. Equal param/data/step counts but **~9% training CPU gap**; do not claim equal training FLOPs or time. No causal unshown-action outcome, independent receipt verification, world-model safety, real commercial decision utility or System-2 claim.

All previous TAC master-plan positive and negative findings retained, no SSM conversion or automatic merge.
