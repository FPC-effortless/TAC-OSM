# ME017 — Frozen real-history credit assignment and native slow-memory audit

Status: preregistered diagnostic. Does not open a fresh cohort or train a new model. Preserve ME015 chrono+causal TAC as the default *research candidate*, not the negative ME016 identity slow path. Do not merge or make commercial claims.

## Scientific rationale
ME016F froze 12 original 600-step 135,050-active-parameter checkpoints. Native engineered TAC beat original and anchored GRU in 3/3 paired seeds on genuine never-trained user3 responses, but the new direct fast-to-slow identity inside tanh lost to original native TAC in 2/3 seeds. Original parameter count, data and optimizer updates are equal; actual training FLOPs and CPU time were not equal. This is not a fully compute-matched architectural victory.

Question: does learned recurrence add genuine history sensitivity beyond the explicit analytical prior32 full-watch rate, and are useful gradients reaching native fast and slow state over 32 historical observations? Bounded state is not an information-retention proof.

## Frozen source boundaries
1. Restore the original me016-original-real-innovation-tac-strong-gru archive from run 39412071932, commit e12982eb59a2b820855888b2df0f74d2553abb40. Check unmodified original results JSON SHA256 6870e254b65b74e8a55a242a3528aa478bf7bd08c6812ce92bf32a87fd4ba416, plus all 12 weight digests and metadata BEFORE reading any real data.
2. Authenticate genuine original KuaiRec ZIP MD5 261550d472c48eff4990fb13c0e5bcf7, SHA256 c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c. Only original FIT user_id%16==0 event targets dated BEFORE Aug 1 2020 are analyzed. The data reader skips rows on/after Aug 14 before response parsing; discard previously opened Aug01–Aug14 DEV for all ME017 measures.
3. Exactly 384 equally spaced real FIT windows, fixed positional selection before diagnostics. The 12 checkpoints use identical actual 32 prior actions/outcomes and current known action, genuine labels for diagnostic Brier, and no optimizer steps.
4. Never re-open user1/user2/user3 future labels. The experiment is NOT independent confirmation and no model promotion follows from it.

## Exact differentiable measurements
Let L_i be the current-action logit, x_ij the j-th historical exposure/outcome embedding+metadata, z_ij native fast state, and s_ij native slow state. Measure G_x(j) = mean_i ||dL_i/dx_ij||_2, G_z(j)=mean_i||dL_i/dz_ij||_2 and G_s(j)=mean_i||dL_i/ds_ij||_2 on genuine FIT histories. Publish all 32 lags and ratios of oldest/newest magnitudes. x is explicitly a separate differentiable leaf to isolate the neural recurrent pathway from the analytical prior32 logit. These are logit sensitivities, NOT causal values, guaranteed Jacobian singular values, or verified long-term-memory utility.

For native TAC, reconstruct original output exactly then set final slow-state to zero (and in a separate experiment set final fast-state to zero) only at readout. Report probability movement and Brier difference on same FIT labels. These artificial internal counterfactual forward passes are ONLY controlled attribution diagnostics; they are not real interventions or deployable performance evidence. Include actual mean fast/slow gates, all seed and arm results, original training CPU times and gradient calculation time. For GRU, measure corresponding x gradients, not invented GRU slow-state metrics. Fail closed on mismatch with original logits.

## Fixed interpretation rules
A. If oldest/newest input gradient ratio for native engineered TAC is below 0.10 in at least 2/3 original seeds, record genuine FIT evidence of local 32-step sensitivity imbalance, NOT proof of catastrophic forgetting. Do not retune this threshold after seeing data.
B. If zeroing native slow readout gives nonpositive FIT Brier change in >=2/3 original TAC seeds, the slow representation has not shown necessity on this in-sample test. Do not delete it without independent frozen real confirmation.
C. Compare ME016 identity gradient versus original, but never promote the identity change on gradient grounds: it already failed its independent Brier improvement gate.
D. Never call history-length-32 gradients a verified test of long-history reliability, future multi-environment state accuracy, causal utility, or a fully matched compute-budget win.

## Next minimal mathematical correction, conditional on ME017 diagnosis
The prior ME016 direct inside-tanh identity is not default. An optional bounded proposal to investigate is:

    p_s = tanh(W_s[z_t;s_(t-1)])
    s_t = (1-g_s)*s_(t-1) + g_s*((1-beta)*p_s + beta*z_t)
    beta = 0.25  (fixed, not trainable)

With 0<=g_s,beta<=1 and each prior-state/proposal/fast-state coordinate in [-1,1], convexity guarantees the next slow-state stays in [-1,1]. At beta=0 it reduces EXACTLY to the successful ME015 slow update. Holding the gate and learned proposal constant, a direct partial beta*g_s*I path exists from z to the slow proposal; the full Jacobian may still vanish or cancel. Do not claim a nonvanishing-gradient theorem. This is not a GRU/SSM replacement.

No training, superiority, or promotion claims are authorized by numerical preflight of this candidate. A separate future trial must preregister real data, strictly matched active parameters, training data, optimizer updates, seeds, measured compute/time budgets and independent later real user/domain test. Already exposed user1/2/3 cannot be presented as new confirmation.

## Evidence boundary
Logged KuaiRec offered videos and watch outcomes do not supply randomized policies, trustworthy propensities, externally verified receipts, true intervention costs or identified treatment effects. These experiments support factual action-conditioned prediction only. Every prior ME001–ME016 result, failed gate, real-only evidence requirement, and matched-parameter/data/compute comparison rule is preserved.
