# COLLECT014 — trained routing and read/write-order repairs

This 2×2 internal repair pilot reuses all three immutable COLLECT012 hard-hybrid
controls and trains three new settings, each with seeds 12001/12002/12003:

| Setting | Phase read/write order | Router balance loss |
| --- | --- | --- |
| Frozen control | Read then write | None |
| balance | Read then write | 0.01 × mean-layer Switch loss |
| write_first | Write then read | None |
| combined | Write then read | 0.01 × mean-layer Switch loss |

All graphs have 99,971 parameters and identical initial parameter tensors per
seed. The balance-only forward/state and language-only gradients match the old
graph exactly in the executed FP64 fixture. No reader, parser, support filter,
extra parameter or synthetic training target is added. The new read order
restores current-value credit in actual whole-graph fixtures; its trained
benefit is not assumed. Balance is a training loss, not a batch-global inference
route, and it cannot establish useful retrieval merely by evening bank counts.

Original Tiny Shakespeare TRAIN/DEV bytes, 1,024 FP32 AdamW updates, batch 4,
length 128, learning rate .0003, betas .9/.95, weight decay .1 and clipping 1
match COLLECT012. Fixed final checkpoints are used, not best-DEV selection.
There are nine new full fits, separate from three new eight-update smokes and
the three reused full controls. All new model/optimizer/RNG snapshots at
0/256/512/768/1024, per-step loss/auxiliary/routing journals and replay evidence
are retained. Each fit presents 524,288 TRAIN bytes with replacement, an early
learnability budget rather than convergence.

This is a same-graph factorial repair pilot. Complete training FLOPs and
independent qualification remain unmeasured; equal parameter/token counts do
not qualify architecture superiority. The standing operator internal_research
amendment is used unchanged through the pinned COLLECT012 policy. TEST bytes
are absent. No full formal/IEEE/AD release, promotion or global-attention
replacement is claimed.

`EVALUATION_PROTOCOL.json` freezes three diagnostic groups before fitting:

1. The same 8,192 exposed-DEV targets used by C12, with phase-read zero/shuffle
   interventions, all-bank usage, and finite 512-byte generation at .8/1.2.
2. Eight original 512-byte DEV continuation windows, comparing 128-byte chunk
   carry, reset and wrong-history states.
3. Twelve shared generated histories with four active keys, two overwrites,
   four present and two absent queries, and 64/256/1024 distractor bytes. Query
   answers are excluded; query branches cannot mutate the common prefill state.
   This is a zero-shot unfamiliar grammar, scored with nine fixed candidates
   without threshold tuning, alongside full-vocabulary predictions/probability.
   Floor performance is possible and must be retained. These fixtures do not
   establish trained retrieval utility or production absent-key rejection.

All histories/windows are reused across models/seeds; repetitions do not
multiply independent evaluation data. Removing all state and substituting wrong
history are distinct controls. Per-seed language and retrieval outcomes matter;
bank usage alone is not success. Reused controls remain byte-for-byte original
and are newly evaluated into separate directories.

The CPU route retains max parallelism three and 45-minute job limits. Local
eight-update smokes completed with changed weights and exact saved replay;
their measured ~1.0–1.2 seconds/update projects each fit within the existing
bounded CPU policy. No GPU quota is used. The workflow runs only on a change to
the explicit launch request, so documentation/results publication does not
restart fits.

Lineage: COLLECT011 runtime and COLLECT012 fit commit
036d778255a416d3aae886acdbfeef8fb1b6a1ab; parent research commit
c78c77a654571e64237c68ce046ea4a2019a4608. The exact preconditioner is unchanged;
it is the normalized write, not exact RLS. Whole-graph independent-expression
FP32/FP64 logits/state/gradient, masks, causality, chunk continuation and owner
isolation fixtures run before fitting. This is executed finite-fixture evidence,
not mathematical qualification.

The user's residual-tiered proposal is recorded and tested separately under
`residual_tiered/`. It does not modify these registered training settings.
