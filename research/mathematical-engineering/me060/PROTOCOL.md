# ME-060 — original FIT gradient connectivity and effective active-parameter audit

Status: source-only diagnostic, **not a trained capability experiment**.

## Motivating problem
The nominal parity count of 38,266 native and 38,295 GRU trainable parameters is based on `requires_grad=True`. This does not establish that those parameters receive actual gradients in the registered ME-059 model training. Both candidate and GRU process the entire 256 observed-frame burn-in under `torch.no_grad()`; the gradient-bearing horizon only calls observation and prediction, not verification. Fast/slow retention, action-feedback verification and GRU feedback modules may therefore be effectively frozen.

## Independent source-backed diagnostic
- Reuse the official pinned independent robot task and *all 228 original FIT file hashes*, SHA256 `c4064651e91cced7aafc11337c32a27266f470ea10388bb0dd2562ae955f88fd`.
- Use ME-059's immutable genuine FIT physical prior and seed 58001's first **exactly equal original FIT paired window**.
- Run the **exact current ME-059 native and GRU training losses** (native pre-feedback 128-transition dense auxiliary plus 16-step rollout, GRU 16-step rollout), with one backward pass per architecture.
- For every named trainable parameter, record element count, whether `grad` exists, whether its entries are finite, whether any are nonzero, max absolute gradient, and whether the discarded original FIT AdamW update changes the parameter bytes.
- Compare **effectively gradient-connected** trainable parameter counts, not just nominal `requires_grad` counts. One sample can have incidental zero gradients; structurally absent `grad=None` due no-gradient code path is stronger evidence.
- No original DEV/TEST physical data opened, no scored trained checkpoints, no predictive superiority or policy claim.
- The pilot update is explicitly counted as an original FIT *discarded diagnostic*, not full optimizer training.

## Consequences of a failed reachability audit
Parameter parity cannot be assumed from `requires_grad` alone when large feedback modules never receive learning signals. The next integrated architectural repair should train genuine feedback and retention through observed transitions within an equally budgeted truncated-BPTT window **for both arms**, without inserting post-outcome information into pre-outcome prediction. Reprofile corrected FLOPs and parameters and rerun original FIT qualification. Preserve ME-059 empirical outputs as developmental, rather than erasing negative evidence.

Prior art R2D2 (recurrent burn-in), DreamerV3 (feedback-trained world dynamics), and TD-MPC2 (action-conditioned multistep models) establish relevant training ideas; this study tests their concrete implementation reachability in TAC.
