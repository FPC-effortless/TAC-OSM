# GRAPH015 — preregistered delta-correcting native TAC: four concurrent gates

**Status: frozen engineering proposal before new training.** 2026-10-28. Existing System-1/language family only; no seventh family. Stacked on GRAPH014 #542. Original GRAPH009–014 results and trained bytes must remain untouched.

## Hypothesis and failure conditions
Hypothesis: separating source decay from rank-one error-correcting delta updates can improve TAC's **native** next-byte prediction without needing more than 500K learned parameters. The Gated DeltaNet mechanism is *prior art*, not a TAC invention; this variant tests an adaptation to the existing three 8-key matrix banks and postscan concept/wave head, **not** an implementation of Gated DeltaNet or Mamba-3.

Registered first diagnostic uses exactly 30 authentic Vim9.1 FIT and six **exposed** DEV publisher chapters (36 SHA256s checked against historical source protocol), original seeds 1701–1705, 120 updates each, context128, batch8, original historical per-seed FIT sample schedule, AdamW lr=.001, one-step next-char categorical CE, gradient clipping norm1. Original historical TAC 499290 and GRAPH014 SSM 498288 frozen reference are *different-fit checkpoints*, not a compute-equal prospective baseline. Only call this **exploratory exposed-DEV** even if numbers improve. Report every seed, all document scores, native NLL/Brier/accuracy when available, and any regressions. No protected TEST or user-source document acquisition. Full training objective for each prior model and warmup/scheduler provenance must be checked separately; a matching sample digest is necessary but not sufficient.

## Proposed recurrence (independent 8x8 native and two 16x8 source-aligned banks)
For input-only normalized k_t, bounded v_t, and coefficients from only the currently available token:
```
rho_t = .995 + .004 sigmoid(alpha_logit_t)
beta_t = rho_t (.05 + .9 write_gate_t (.5 + .5 erase_gate_t))
M_t = rho_t M_(t-1) + beta_t (v_t - M_(t-1) k_t) k_t^T
    = M_(t-1) (rho_t I - beta_t k_t k_t^T) + beta_t v_t k_t^T.
```
The learned active parameter count remains **499290** exactly; all original gate rows remain connected. Write-mask=false is identity/no-write; original prefix causality and read-before-write remain unchanged. O(log T) parallel-depth right-affine composition preserves chronological updates in exact arithmetic, with O(T log T) work for current Hillis–Steele implementation. FP32 roundoff and complete AD/Lean refinement are open. Because `0<beta<=rho<.999`, and `||k||<=1`, transition matrix P is positive semidefinite with operator norm at most rho. This yields a conditional stability/finite-horizon bound, **not** evidence of learned memory or correct long-history retrieval. Slow/fast banks are learned persistent state, not a guarantee of useful retention.

## Falsifiers
- Exact 499290 learned parameters; every parameter gradient exists and finite.
- Rank-one keyed residual correction numerically matches the algebra; spectral P norm ≤rho.
- Parallel and chronological scan output/gradient correspondence; source-causal no-future and masked admission equivalence.
- Inspect real training numerical stability, reject NaN/nonfinite, archived source mismatch, FIT sampling mismatch, original model-score drift, unauthorized source boundary.
- Compare five-seed *native* exposed-DEV NLL against old TAC and fresh GRAPH014 SSM. Report losses even if candidate fails; pre-cache main criterion for retrospective diagnosis only.
- Equal-cache diagnostic may be run **only** after training with the *same* shared source-only count/query posteriors. Preserve negative learned-query transfer effect and scorer's scoring-only status.
- One-step operator profile for TAC old/new/SSM on equal [8,128] with declared coverage; record operator FLOP ratios but **do not certify training-budget parity from incomplete profile**.
- Never claim recurrent algebra alone caused a model gap: optimization, learned representations, auxiliary losses, model architecture, trace exposure, and FLOPs also differ.

## Four concurrent workstreams with honest gates
1. **Native math/model**: source-pinned module + numerical regressions and genuine real FIT training, checkpoint original bytes, local clean-load score replay; negative accepted.
2. **Compute-parity certification**: implement executable fail-closed active-parameter, exact FIT sample digest, total training FLOPs/profile coverage and objectives checks. Missing operator coverage or receipts ⇒ FAIL/UNKNOWN; no claimed fair winner.
3. **Checkpoint/Git replay**: register deterministic original SHA-256 and cold-replay code. Original .pt must actually be Git/LFS-tracked and lawful source bytes independently verifiable. A sandbox ZIP, GitHub CI tests and SHA-only receipts do NOT meet REPO_REPLAY=PASS. No attempts to bypass source license.
4. **Fresh real TEST**: prospective acquisition and source authorization/data chronology, fully new documents previously unviewed by design and scoring, frozen trial and primary metric before opening. Need genuinely unopened source identities, not old Vim or GRAPH011 Python files. Do not access until independent receipt/availability is established.

## Evidence thresholds and exclusions
A native NLL decrease on **exposed DEV** means provisional progress only. Independent model-skill gain requires preregistered meaningful reduction with case/document clustered 95% CI and failure guardrails on genuinely new authorized real documents, with all five seeds and no worse meaningful reliability or cost failures. Until then: TRAINED=diagnostic only, INDEPENDENT_TEST=NOT_RUN, FLOP_PARITY=UNVERIFIED, ORIGINAL_REPO_REPLAY=BLOCKED, ARCHITECTURE_SUPERIORITY=NOT_ESTABLISHED, CUSTOMER_VALUE=UNKNOWN.

## Contemporary sources
Gated Delta Networks (Yang et al. 2025), https://research.nvidia.com/publication/2025-04_gated-delta-networks-improving-mamba2-delta-rule ; Mamba-2 SSD (Dao and Gu 2024) https://proceedings.mlr.press/v235/dao24a.html ; Mamba-3 (Lahoti et al. ICLR 2026) https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html. Those external architectures have substantially more engineering than this candidate; this is not a direct replication.
