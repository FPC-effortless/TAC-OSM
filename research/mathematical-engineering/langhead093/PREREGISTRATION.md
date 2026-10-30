# LANGHEAD093 — retrain direct tied embedding output with new source-document owners

**Preregistered 2026-10-30 before optimizer steps.** Stack on original frozen-checkpoint mathematical result HYBGEOM092 PR #643. Preserve all original source and old trained checkpoint files; draft and do not merge. The 25 model trial is internal exploratory research, not an architecture-superiority qualification.

## Hypothesis

The earlier apparent fixed Euclidean gain is mainly a direct tied current-byte-to-candidate embedding similarity head, not a learned geometric manifold or improvement in native memory. Falsification requires a fresh independently trained tied-head model to reduce mean next-byte negative log likelihood relative to the separately trained original TAC on preselected, different real documents. A candidate-permuted head should fail to reproduce the gain. A norm-only head distinguishes dot from norm effects.

## Model arms and parameter gate

Five arms share exactly the original 100,521p CompactDualClippedTAC086 recurrent graph, with fast/slow 8x8 memories, aligned associative writes, waves, concept and upper state, all unchanged: (1) original 100,521p; (2) tied_dot 100,522p; (3) tied_dot_norm 100,522p; (4) norm_only 100,522p; (5) shuffled_dot 100,522p. The four modified arms add one actually learned output gain scalar. The shuffle is a fixed candidate bijection, index=(73*y+41)%256; not based on target labels.

For observed byte embedding e_x, candidate e_y and gain g=0.02 sigmoid(raw_gain), the centered logit update is: tied dot g*2(e_x dot e_y), full g*(2e_x dot e_y - ||e_y||^2), norm-only -g*||e_y||^2, or candidate-shuffled dot g*2(e_x dot e_perm(y)). Mean centering does not change softmax. Only the current observed input token and trainable embedding are used. No retrospective evaluator-selected addresses or future tokens enter prediction.

## Exact train and new evaluation cohorts

All 25 models fit from scratch on genuine PSF CPython v3.13 publisher 14 original FIT documents plus UTF8 supplement, registered aggregate FIT SHA256 911981cc63fd5436e0495cb067784e53475971131c811b7026e07d97a886d2d4. Seeds 1701,1702,1703,1704,1705. Exactly same deterministic sampled FIT owner/start/byte sequence across arms of the same seed; AdamW learning-rate .001 weight decay .01, global clip 1, no tuning, 192 updates * batch4 * context64 = 49152 true next-byte targets per model. New heads have active trainable gradients. Same parameter count to within 0.001%, but **extra decoder FLOPs are not matched**. Neither equal wall time nor equal token count qualifies a strict compute-matched architectural win.

New source-owner cohort, deliberately different from earlier exposed CPython DEV owners: Doc/library/zlib.rst (Git blob 965b82a3daffb921a1b06cdc4ebbd324965f2f75), Doc/library/bisect.rst (78da563397b6255139f23d03502867cdcee384fc), Doc/library/tomllib.rst (521a7a17fb3e8bc9984e0d875ba7a3c89de231d8), pinned exact CPython publisher commit 60403a5409ff2c3f3b07dd2ca91a7a3e096839c7. Metadata fixed before any model was trained or evaluated on those sources. They are three new documents **from the same publisher**, not three independent research environments. Do not call it cross-publisher generalization or untouched TEST.

The new bytes are downloaded and their Git blob hashes checked **only after each seed model checkpoint has been sealed and cold loaded**. After 64 byte burn-in, score ordinal t=64..511 (448 targets per owner), giving 1344 genuine next-byte scores per seed and 6720 per arm. Source sha256/bytes and every per-byte target result are retained. No hyperparameter selection based on this evaluation; protected TEST is never accessed.

## Replay and decision

Require repo preflight of this explicit 25-arm prospective single-model registration, phone lab CPU placement, actual source digest, 100k parameter measurements before first optimizer step, same sampler replay per seed and nonzero head gradients. Before any DEV read, save full original model optimizer RNG checkpoint. After fit, independently cold replay all original publisher source bytes, trained weights, optimizer steps, and **all** per-byte new DEV scores from the exact original checked-in checkpoints. Fail shut on missing artifact or wrong SHA; preserve negative seeds and source owners. Calculate paired seed-level mean NLL differences with 95% t intervals, accuracy, per-owner tables, latency and full training receipts; do not multiply sample size by pseudoreplicating five training seeds over the same three source documents.

No GitHub merge, Kaggle T4, paid API budget, protected TEST, independent claim or checkpoint promotion. Remaining OPEN: matched-flop GRU/SSM/Transformer training, second publisher, physical elapsed-time adaptation, micro/macro actual outcomes, full PINN/PNDS source graph Lean+Python proof.

**Research classification:** internal_research; single_model_training; no_architecture_superiority_claim=true. Output may support a next experiment, not general capability or safety certification.
