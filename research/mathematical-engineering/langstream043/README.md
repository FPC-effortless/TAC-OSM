# LANGSTREAM043 — same-model persistent inference execution

Implemented on the original trained040 graph-free48D499623p/FFN455 TAC; no new
learned weights. Source first-stage concepts/associative/wave computation and
upper recurrent read/write remain. This is an inference-only API, not a new
language capability, training method or formal safety/IEEE/AD qualification.

```
runner = IncrementalTAC043(original_loaded_model.eval(), 'cap_all')
logits, state = runner.prefill(prompt_bytes, owners=('document/window-0',))
next_byte = logits.argmax(-1)
logits, state = runner.step(next_byte, state, owners=('document/window-0',))
runner.save_state(state, new_snapshot_path)
state = runner.load_state(new_snapshot_path, owners=('document/window-0',))
# state.tensors['last_logits'] retains the predictive distribution on restart.
```

Owners identify distinct streams, not model features. All batch rows share one
observation clock; asynchronous lengths require separate states. Call `step`
to continue an existing state. `prefill` initializes a new reset prefix.
Native and042 cap_all positional profiles are supported separately; source
clock, ordinal keys and context-mean denominator always advance correctly.
The admitted write mask affects native/aligned banks only, not all observations.

## Exact semantic target

Compare with the ORIGINAL model evaluated on the SAME complete growing prefix.
This is not equivalent to dropping tokens from a128-byte rolling window and
resetting the original model's clock/state. Native postwrite read and shifted
historical key, aligned prewrite reads, old wave/concept state, basis running
sum, three lexical lag embeddings, prior context key and upper48D state are
carried explicitly. Fast/slow88D features remain per-token computations, not
new independently persistent states.

Bulk prefill executes the original batch model once, recovers its prewrite
states and upper prior, and computes the last update. This adds some prefill
cost but avoids a slow serial warmup. Subsequent byte updates process no entire
prefix and retain no token history. Original files/weights are untouched.

## State accounting and guards

| Tensor | Coordinates/stream |
|---|---:|
| Native8×8 |64|
| Aligned2×16×8 |256|
| Wave8×2 |16|
| Concept6×16 |96|
| Basis sum |88|
| Lexical lag3×48 |144|
| Previous native key |8|
| Upper recurrence |48|
| Cached next-byte logits |256|
| **Total** |**976**|

FP32 retained tensor payload3904bytes/owner; FP64 doubles it. This excludes
weights, tensor/Python object overhead, bounded owner/clock/version metadata,
caller input/output/source, temporary matrices, bulk scan workspace and runtime.
It is not total device RAM or a raw-history recovery guarantee. Prefill working
memory still grows with prompt length. Clock is int64-bounded; batch<=64;
owner identifiers are explicit bounded strings. Only batches1/2 and registered
finite prefixes were empirically qualified in this increment.

Transitions do not mutate the caller's tensors. Model weights, device, dtype,
registered parameter versions, eval mode and unregistered hooks are checked.
Snapshots bind model tensor hash, source/runtime profile, owner, clock, mode,
shapes and dtype, use a corruption digest, reject overwrites and load with
weights_only=True. These are consistency guards, not independent signatures or
security certification. Unsafe direct `.data`/monkey-patching outside the model
API is unsupported; original checkpoint hashes and final tensor equality are
also checked in the measured pipeline. Snapshot bitwise continuation was tested
on the same CPU/runtime; universal cross-device bitwise identity is not claimed.

The source-reviewed ResearchProbeAdapter exposes all nine real tensor fields;
forecast consumes a separate current observed query byte into a NEW state and
leaves the caller's old state unchanged. Autograd through the inference API is
intentionally disabled; its AD probe is INCONCLUSIVE rather than PASS.

## Evidence / remaining work

See research/results/LANGSTREAM043_RESULTS.md and full JSON/PARITY.jsonl.
Prereg ec638bb5b800083929be8c13c79c181cc5c980e0 before implementation/testing,
CPU20min route, metadata gate PASS, no GPU/optimizer/DEV/TEST metrics. Original
five040 checkpoints:300 real-prefix cases, original full graph and independent
PNDS functional step, all reported states/logits within frozen FP32/64 bounds.
Seven local tests include masks, lengths1/7/17/129/513, batch/owner isolation,
bulk/serial/chunks, source/model gates and bitwise snapshot restart. Initial
adapter import collision is preserved in INITIAL_CHECK_HISTORY.json.

[PNDS66](https://github.com/FPC-effortless/pnds-lean/pull/66) defines the complete
exact-real token graph, actual455 decoder/48 upper, and independent Python with
no TAC/scans/traces imports. Six scoped checks cover prefix causality, native
write boundaries, declared coordinate count and exact-real affine composition.
The totalized CED inverse/source bridge, global IEEE error bounds and formal AD
remain OPEN. Small numerical reassociation errors do not change the tested greedy
continuations but are not a universal token-ranking theorem.

No mobile/hardware deployment, KV-cached Transformer/BDH/SSM ranking, trained
long-history recovery, independent qualification or checkpoint promotion. All
025/035–042/H64 negative findings remain. Next benchmark must use comparably
cached competitors and identical history semantics; a separate retrieval study
must demonstrate useful source information rather than infer it from faster
execution. Customer inference stays offline on user-controlled devices.
