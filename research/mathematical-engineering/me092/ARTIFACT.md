# ME092 original restore and capability contract

Use the same pinned public CC BY4.0 BodenAI/RW-RL numeric source and exact
observed pixel bundle as ME091. No new source roles, future images or commands.
All five final originals, optimizer/RNG, source normalizers, guarded2048step
receipts and sampling digests belong in`research/replay-bundles/ME092/` before
REPORT scoring. Do not regenerate failed models, choose best seeds or ensemble.

Initialization is explicitly two-donor, fixed by index acrossALL5seeds:
94907native values fromME087 and7792visual values fromME091,1526location values
zeroed. New2048AdamWupdates;10240total pipeline steps including parent work,
6144updates along each retained native/visual value's lineage,2048for location.
The single deployed learned model is104225parameters; donor models are only
initializers, never additional inference experts or frozen encoder loopholes.

One shared forward/one batched3task VJP, PCGrad,clip1,one guarded optimizer step.
Actual CPU time is reported, not a matching gate or FLOP-parity claim. The raw
log preserves a legacy`.jsonl`stdout filename with a plain completion marker.
Its `tasks:3` counter overrides a same-named component-loss list in the original
logging dictionary; component loss curves are NOT available in that log. Raw
loss, gradient norms and projection counts are original, not corrected/replaced.
The source, original sampler/RNG and final optimizer are preserved exactly.

After all originals are Git-frozen, reproduce with no fitting:

```
python scripts/me091_restore_pixels.py
PYTHONPATH=src:. python scripts/me092_train_world.py finalize --study me092
PYTHONPATH=src:. python scripts/me092_train_world.py replay --study me092 --cache /tmp/numeric-source
PYTHONPATH=src:. python scripts/me092_cross_cpu_audit.py --study me092 --cache /tmp/numeric-source
```

Frozen original float32 values are promoted into the normative float64 full
forward. Original rtol1e-5/atol1e-6, source observations and quantile iterations
remain fixed. Independent restoration covers all27600forecast records; first-
original normalized/physical-unit compiled programs are separate fixed-profile
execution evidence, not entireREPORTcompiled qualification or learned ability.

The fixed full joint gate remains compulsory. Additional99.54545%interval
covers11declared recipes only; adaptive exposedDEV is not independent proof.
Forty recording episodes,960registered H1origins,956H4/928H16, not every possible
source frame. Seed/horizon repeats and overlapping windows add no independent
operator/site evidence. No autonomous action, enterprise, customer-value or
commercial qualification is unlocked by software/numerical replay alone.
