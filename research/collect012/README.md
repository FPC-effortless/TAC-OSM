# COLLECT012 — genuine text learnability, not architecture qualification

This continuation preserves COLLECT011/PR #697 unchanged and asks a narrower,
executable question: do the actual attention-reference, wide phase/P and
discrete-hybrid/P TAC graphs learn ordinary next-byte text without synthetic
key/role labels, a grammar parser, or the support filter?

The standing operator amendment dated 2026-10-27 permits `internal_research`
on genuine public FIT/DEV without independent cryptographic signers. Its exact
source commit/blob is preserved in `OPERATOR_AMENDMENT_PROVENANCE.json`.
It does not authorize protected TEST, qualification or checkpoint promotion.
The default branch's older blanket signer requirement was too broad for this
ordinary internal scope. No qualification launcher or historical gate is edited.

Each graph is a separate single-graph learnability registration, with three
fixed seeds and 1,024 FP32 AdamW updates, batch 4 and 128-byte context. Model
counts are 99,875 / 99,879 / 99,971, all inside 95K–105K and within 0.1%.
Matched counts, windows and update budgets do not establish matched FLOPs:
these fits cannot rank architectures or establish superiority. The separate
compute-matched comparison gate remains OPEN. No fit is disguised as a no-fit
diagnostic; the eight-update optimizer smoke is separately registered training.

The publisher's pinned Tiny Shakespeare repository identifies the data as a
subset of Shakespeare and declares MIT in its pinned Readme. A byte-range HTTP
206 request independently recovered exactly the first 1,059,624 source bytes:
TRAIN and DEV hashes match the preserved originals. TEST bytes were not read.
`data/FIT_DEV_SOURCE.zip` contains those two permitted splits only; its checksum
and explicit member list are checked before training. No examples are generated.

The 99,879-parameter phase model's first actual CPU optimizer smoke completed
eight updates in about 15 seconds, with changed weights, correct AdamW counters
and exact serialized-checkpoint replay. This is not a trained language-capability
result. Its measured 1.68 seconds/update projects a full fit to about 29 minutes,
inside the policy's 45-minute CPU route. Full fits are planned on bounded GitHub
CPU workers. No Kaggle GPU is submitted by this study; Kaggle authentication was
verified read-only. The existing protected two-T4 route remains required for any
future GPU allocation.

Full graph equations are independently re-expressed in `collect012/reference.py`
and the inherited native references. FP32/FP64 logits, carried state and all
parameter gradients are checked against the actual 100K source graphs, along
with masks, causality and chunk continuation. These are executed finite-fixture
checks, not a complete IEEE/AD proof. Lean/native full formal release and reciprocal
PNDS mathematical qualification remain OPEN and are never inferred from training.

Every full fit retains initialization, immutable 256-step model/optimizer/RNG
checkpoints, a hash-chained per-step journal, original protocol/source identities,
final-model replay input/logits and honest completed/attempted counters. A timeout
or exception is partial/failed, never a complete model. A separate fresh process
verifies the final checkpoint and fixed exposed-DEV scores, postfit memory-read
knockouts/shuffles, and 512-token generation at temperatures 0.8 and 1.2.
These do not establish coherent long-horizon language or exact semantic binding.

Run from this directory:

```bash
python -m collect012.restore_data
python -m unittest collect012.test_prefit -v
python -m collect012.train --arm phase_p --seed 12001 --output results/phase_p_12001
python -m collect012.evaluate --run results/phase_p_12001
```

The named launch workflow runs only when its explicit launch-request file changes.
Documentation/result publication does not rerun training. Qualified research,
protected evaluation, full FLOPs matching and complete mathematical release are
separate unfinished obligations. All earlier failed gates and negative findings
remain historical evidence.
