# LANGLAYER041 — true original text longer-prefix functional test (NO FIT)

This prospectively registered **no-training** study reads the 20
immutable actual trained original source checkpoints from LANGLAYER035
(48D upper graph), LANGLAYER039 (59D graph-free) and LANGLAYER040
(48D graph-free,59D upper graph), five seeds per configuration.
No checkpoint reinitialization/retraining, optimizer updates or
DEV-based hyperparameter selection. All original source identities and
checkpoint SHA256 digests must match frozen original receipts.

## What "long context" really means here

Original CONTEXT026 first stage and all upper variants are **window-reset**
and have NO externally carried state. 512 observed original bytes
is a longer **within-window prefix**, not a 512-token persistent stream.

The same 8 target offsets from each of four authentic already-exposed
PSF CPython3.13.0 DEV publisher documents are scored at L128,256,512,
for **32 identical byte targets per model/seed**. Anchors are the eight
integer-spaced real source positions between byte1280 and file end-2.
Every target's true immediately preceding original 64 bytes are
unchanged, both in genuine and donor conditions.

For prefix length L:
- Genuine: exact previous L source bytes from the target's document.
- Donor: first (L-64) authentic **earlier** bytes of that *same*
  publisher document followed by exact true last64 bytes.
- Thus both conditions have the same immediate local context, local
  model time position, target labels and source identity; only
  more distant historical content differs. Donor bytes cannot include
  target/future information; no document concatenation or TEST.
- Report NLL(donor) minus NLL(genuine). Positive is evidence that
  the model's present predictions benefited from historically
  correct **observed earlier-prefix content** under this intervention.
  Negative means wrong/far donor content helps or true older history
  hurts under the specific train protocol.

Score same exact 32 target offsets for all lengths, seeds, arms. This
is a narrow, previously exposed DEV diagnostic. Context length changes
original positional embeddings and network computation length: any
L128→L512 score difference **alone** is NOT a causal memory finding.
Compare donor vs genuine at identical length instead; still no
independent qualification or source-generalization claim. 32 target
bytes/seed are too few to establish robust long-range language skill.
The donor prefix begins at the first byte of the same source, not a
probabilistically matched alternative natural passage; artifact
boundaries must be discussed as potential controls.

## Research evidence limits

- Full checkpoint, AdamW optimizer/RNG and source-order SHA validation.
- 4 arms×5 frozen seeds×3 lengths×2 source conditions, no new fits.
- CPU-only original genuine publisher material. All 20 evaluations
  and independent source/target identity aggregation required.
- No complete composed full-model PNDS Lean, IEEE/AD proof,
  independent TEST, streaming carry interface, long-range functional
  superiority, equal training-FLOP claims, model promotion or merge.
- Factorial 040 is a parameter-budget reallocation with variable
  final FFN width, not fixed-decoder pure graph causality.
