# ME020 — Preregistered genuine frozen-TAC 128-event mathematical mechanism audit

## Exact problem and empirical baseline

ME019 original authentic KuaiRec previously inspected DEV (18 fitted original models) showed native TAC 32-event mean Brier 0.203345871 vs 128-event 0.204557548 (worse all three seeds), original GRU 0.204369427 vs 0.203677189 (better three). ME019F original once-frozen first future/new-user4 exposure (43835 actual video impressions, 442 users) similarly found native TAC means 0.197458307 (T32), 0.198100030 (T128), worsening 2/3; original GRU 0.197790492 versus 0.196809072, improving 3/3. Future independent test evidence is sealed and **may not be retuned or re-opened as if fresh**. Training model sizes exactly 135050 and target data/update budgets matched but original CPU/FLOP budgets NOT equal. No fully controlled architectural win claim.

**ME020 question:** what is *measurably* happening inside original, unchanged, SHA-verified TAC over 128 REAL observed past actions/reactions? Distinguish dynamic propagation/credit problems, nonlinear gate activity, and whether old actual records are decodable and relevant. DO NOT add a new decay, attention, Titans/MIRAS controller, memory reset or SSM conversion in this experiment.

## Immutable originals and source

- Original ME019 successful GitHub Actions **39443325909**, archived original raw \`results/me019/results.json\` SHA256 **2a10ae3205e866ed792f5ad145e3c3c775e95c39de675f421184de44e56455b6**, 18 trained original model checkpoint digests from this manifest, **ALL HASH-VERIFIED BEFORE READING SOURCE**. Original frozen network consists only of native chrono+anchor T32/T128 (3 seeds each) and original GRU T128 (3 seeds), 9 measured models. The other nine original checkpoint files are SHA-verified but not refitted or instrumented.
- Original publisher KuaiRec \`big_matrix\` ZIP MD5 261550d472c48eff4990fb13c0e5bcf7; SHA256 c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c.
- **ONLY original early FIT uid%16==0, timestamps <2020-08-01**, original ME019 genuine H128 eligible target rows, source-only sampled 10776 full FIT anchors. DO NOT read historical Aug01–Aug14 DEV, later Aug14–Sep06, user%16 1/2/3/4 or future test responses.
- For fixed source-only sampled 1200 original FIT windows, compute actual real 128-event observation histories, observed actions, feedback, elapsed-time and actual current action/response. First 72 input-gradient/gate windows, first 12 exact full-Jacobian windows, all 1200 for frozen linear probes. These are **in-sample original FIT diagnostics** and user-disjoint *probe* evaluation within FIT, not new independent prediction evidence.
- Every frozen model remains unchanged; no network optimizer updates. ME020 probe decoders are **separate fixed ridge regressions trained only on early FIT**, never inserted into native model or evaluated on independent heldout future cohorts.

## Mathematical diagnostic 1: FULL coupled recurrence Jacobian, not slow-only

Native fast state z, slow state s, input u. At each real chronological event t:

    q=tanh(enc(u)+transition(z))
    a=sigmoid(gate([u,z]))
    z_new=(1-a)*z+a*q
    p=tanh(slow([z_new,s]))
    b=sigmoid(slow_gate([z_new,s]))
    s_new=(1-b)*s+b*p

Let A=∂z_new/∂z, B=∂s_new/∂z_new (partial w.r.t. z_new), C=∂s_new/∂s (partial holding z_new fixed).

    A = diag(1-a)
        + diag(a*(1-q²))*W_transition
        + diag((q-z)*a*(1-a))*W_gate_z

    B = diag(b*(1-p²))*W_slow_z
        + diag((p-s)*b*(1-b))*W_slow_gate_z

    C = diag(1-b)
        + diag(b*(1-p²))*W_slow_s
        + diag((p-s)*b*(1-b))*W_slow_gate_s

    J = [[A,   0],
         [B@A, C]]    shape 64×64

The lower-left *B@A* coupling is essential: \`∂s_new/∂s\` alone cannot diagnose joint information propagation. **Product** \`P_t=J_t @ J_{t-1} @ ... @ J_1\` differs from \`∏_k ||J_k||\`; the latter is at best an upper bound, not true product growth.

- Preflight EXACTLY compares all entries of analytic 64×64 J against numerical PyTorch autograd Jacobian for a constructed sample; artificial sample is numerical-only, never predictive performance.
- On 12 genuine FIT histories x 3 native128 seeds (and native32 separate fits, on most recent real32), use float64 copied original weights for diagnostics only to prevent artificial underflow. Report local largest singular values and cumulative product largest + smallest singular values, and their log10, at registered chronology t=8,32,64,96,128 (where applicable).
- A singular value near zero establishes directional local/cumulative contraction of this exact frozen recurrence on these inputs. It does **not** alone prove loss of useful signal or causal performance failure. Large local spectral norm does NOT prove stable meaningful gradient across time.

## Mathematical diagnostic 2: actual observed gates and neural input credit

- For each original frozen native H32/H128 seed, with exactly 72 genuine same FIT histories, measure per-history-position \`||∂(neural logit)/∂u_t||₂\` (embedding+metadata leaf). The analytic latest32 prior bypass is excluded by design, consistent with ME017; declare that these are not full-model gradients and not causal data value.
- Actual fast and slow gate value distribution by absolute position bins [1–32], [33–64], [65–96], [97–128] (or [1–32] for H32) with quantiles 1/10/50/90/99%, fractions <0.01 and >0.99, and per-position mean z/s norms. Note that gate means, direct carry-only survival ∏(1-g) and saturation fractions are NOT full memory-retention measures and cannot establish overwriting/saturation.
- Report gradient ratios oldest8/newest8; compare with true Jacobian spectra and gate telemetry; never infer mechanism from a single gradient ratio.

## Mathematical diagnostic 3: decodability and actual old-history relevance

Use *only original genuine FIT* user0 histories and the same deterministic 1200 source-only target anchors. Split probe training vs probe testing by **different actual user identities** \`uid%5 != 4\` vs \`uid%5 == 4\` (since all uids%16==0, this partitions genuinely different customers); no overlapping users between regression fit and probe evaluation.

- At early depth 8,32,64,96,128, collect real frozen native128 and original GRU128 learned recurrent states, without detachment/swap in actual inference. Ridge decoder (\`λ=1\` fixed, standardized train-set features, unpenalized intercept) predicts **the genuine binary mean of the first eight past watch outcomes** from those states. Report squared error vs fixed FIT-trained constant mean and observable response-rate control. This measures linear accessibility of facts encoded, not mutual information (we explicitly do **NOT** claim MI has been estimated).
- Separately, use authentic current full-watch outcome on heldout *FIT probe users* to compare a simple real-data ridge predictor based on recent32 response rate vs recent32+older96 rate (same fixed λ, same users and current items/targets). This tests whether **one summary of earlier factual reactions** has incremental observational forecasting utility given the recent rate, not whether original TAC captured all old information.
- The linear probes have different active parameters than original model readouts; they are analysis devices, **not equal-capacity architecture performance comparisons or new model promotion**. A negative old-history increment cannot rule out nonlinear or item-dependent older relevance. A positive increment cannot imply a causal effect or a production action benefit.

## Decision gates (do not retrofit after seeing results)

- **Credit route priority** if actual H128 neural oldest8/newest8 gradient sensitivity markedly declines relative to H32, and exact cumulative Jacobian singular structure also shows strong directional contraction. Do not claim a nonvanishing bound or propose clipping solely based on local J norm.
- **Retention/gate priority** if true fast/slow gates on real H128 inputs clearly approach extremes or change substantially by age, paired with declining state decodability. Gate saturation alone insufficient to diagnose causal overwrite.
- **Representation/relevance priority** if real early-event decodability decays on long histories while historical older96 rate adds FIT-user-disjoint forecasting signal. This suggests testing structurally separated memory, explicitly within TAC, without declaring a solved architecture.
- **Model/optimization mismatch priority** if older info is decodable but neural logit fails to use it. Then first test readout/credit rather than adding gates.
- **No forced structural change:** inconclusive or conflicting diagnostic findings require a narrower genuine-data experiment, not arbitrary selective attention/Titans/MIRAS insertion. Preserve original native 32 T checkpoint and the ME018 1.64× single-thread inference efficiency reference.

## Mandatory scientific limitations

Original trained state is not a policy/counterfactual world model; this source only has factual logged video reactions. ME019 architecture comparison did not match total training wall clock/FLOPs even though trainable params, data examples and optimizer updates matched. No full real-world scientific success, state-of-the-art superiority, commercial readiness or safety benefit claim follows from internal diagnostics. Keep PR #276 and #277 unmerged, all prior failures and evidence intact.
