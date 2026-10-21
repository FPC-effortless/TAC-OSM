# ME039 second construction — immutable inference graph

Registered after the first compiled-core experiment and post hoc deployment
comparison, BEFORE measuring this construction. Retain all prior failed timing
targets and the fact that scripted-core serving was slower than the existing
complete ME038 trace. This is a second engineering build, not a revised result.

Apply torch.jit.freeze(module,optimize_numerics=False) to the ORIGINAL ME038
complete physical-input traced model, eval mode only. Hoist immutable weights,
constant expressions and graph attributes without training, quantization,
pruning, new data, changed state equations or numerics-enabled optimization.
All original learned weights remain as graph constants; zero exposed Parameters
does NOT mean zero learned weights or elimination of model memory.

Require bit-exact original physical-input probe reproduction for all60models,
and equality after save/reload in a fresh Torch-only process. Source archives and
model hashes remain immutable. One backend, no tuning sweep. No new accuracy
scores, closed-test retuning, controller or model-promotion claims.

Primary serving engineering target: geometric-mean speedup>=1.10x over the ORIGINAL
ME038 complete serving exports, on the SAME eight cases (ETTh1/Electricity,
H96/720, two seeds). Three paired repetitions, five warmups/path per repeat and
thirty B1samples/path. Alternate path order each sample/repeat. Single-thread,
no concurrent local CPU-heavy work. Record freezing/setup CPU, per-case/repeat
latencies, cold inclusive first-call cost, file bytes and break-even calls.
Preserve every case and original timing failure; no selecting favorable runs.
All60serialization sizes and exact forecast replay receipts recorded.

This is explicitly a follow-up serving-only optimization after a negative
training/deployment experiment. It is not new architectural novelty, a new
training-speed result or an independent prediction/generalization qualification.
Original full96-step history replay and128float32native state/channel remain.
