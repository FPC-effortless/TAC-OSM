# MLAG-002/003 preregistration: memory role audit and source gradient surgery

Date: 2026-10-28. Pre-analysis: MLAG-002 source-head lesion and FIT-only gradient diagnostics are exploratory but their frozen original/MLAG001 checkpoints are **never trained or changed**. MLAG-003 is a new FIT-only training-objective/optimizer ablation, not a promotion candidate.

## MLAG-002 fixed-source audit

Read 10 original and 10 MLAG001 training checkpoints with SHA256 manifests. Verify original source SHA256, original source chapter FIT/DEV split, 36 real Vim9.1 document SHA256 digests, individual checkpoint hashes, original evaluation metrics and 128-token fixed DEV windows. Evaluate full-next-token vs native-read ablated, aligned-read ablated and both ablated. Independently test historical logits vs zeroed aligned state and off-by-one/off-by-eight address. Every comparison on **the same targets** when shapes permit; address-offset intersection sizes must be recorded. The historical classifier uses source-aligned matrix, not the native bank: a tiny next-token aligned-lesion effect does not establish bank redundancy. Report per-seed/per-gate negatives.

Gradient diagnostic uses **four real FIT** chapters, fixed 128-token windows (not DEV/TEST) to calculate separately the exact PyTorch autograd of CE(next) and CE(history96), and compare vectors on named shared-source, native, history-only, next-head and embedding parameter groups. No fitting. Report vector norms, relative strength, cosine and zero-vs-nonzero gradients.

## MLAG-003 intervention, fixed in advance

Hypothesis: large and occasionally conflicting historical gradients on the shared aligned source encoder might explain low next-token usage despite useful long-delay retrieval. Test a **parameter-neutral gradient-surgery** candidate on the same original learned model (100,680 active params) using the exact MLAG001 objective `L_char+.15*L_h64+.15*L_h96`, **same genuine Vim FIT docs, identical seed initialization, optimizer settings, 120 updates and exactly the same per-seed training sample-order SHA**.

**Only modification**: for learned `rotation`, `source_content`, `source_router`, `source_retention`, `source_strength`, `source_silu_mod`, define `g=grad L_char`, `h=grad (.15 L_h64+.15 L_h96)`. Where `dot(g,h)<0` and `||g||²>1e-20`, replace `h` by `h - dot(g,h)/||g||²*g`. Other gradients untouched; continue existing global clip1 and AdamW. The projection is the **raw gradient's local first-order** non-antagonism, NOT a promise about AdamW loss, finite-precision accuracy or learning the historical target.

Primary exploratory paired metrics: genuine exposed DEV next-character NLL, historical 64/96 NLL, both historical exact-character accuracy, correct/off-key historical NLL and source-aligned lesion. Baseline = archived MLAG001 checkpoints. Report per seed, checkpoint hashes, run-time, projected step fraction. Use original five seeds 1701–1705, **SiLU mode initially**; raw-linear comparator is a separately budgeted follow-up, not quietly pooled. Explicitly account for extra autograd passes, hence NOT equal FLOP or wall-time comparison.

**Decision gate (not inferential TEST):** require <=+0.001 mean next-token NLL versus MLAG001 baseline, <=+0.02 mean history96 NLL, <=+0.02 mean history64 NLL, and at least 4/5 paired seeds non-inferior next-token. If any fail: NO promotion; publish negative outcomes and investigate representation/decoder bottleneck instead. If passed, it is still exploratory on prior inspected DEV; independent new whole-document TEST remains sealed, checkpoint cold CI required.

**Future 256 horizon:** do not train to 256 against original context128 (query invalid). Register fresh context >=320 with a same-context, same-source, same-parameter, same-training-update 64/96 control and a 64/128/256 candidate, complete cost and checkpoint replay. Avoid extrapolating one 96-character result to 256 or 512. Both heads preserved unless a retrained matched substitute shows actual evidence.
