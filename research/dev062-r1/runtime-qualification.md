# R1 same-weight inference qualification

Batched public observation intake preserves all checked predictions and exact
choices on23 restored checkpoints and6784 recorded trajectories per variant.
Float64 thread1 max drift0 (batch),2.85e-14 (batch+cache).

Registered mean total TAC model CPU: base12.897ms,
batched5.761ms,
batched-cache6.315ms.
Batch/base ratio0.4467; cache/equally-batched
ratio1.0962. Batch qualifies locally;
cache does not. The original per-record cache/base ratio is
1.7629, also rejected. These are different forced
traces from DEV062A; don't compare absolute times across recipes as a causal gain.

Includes initial/action-receipt/terminal model observations and choices; excludes
JSON/tensor construction, simulator and verifier. CPU timing on one host is not
production latency, actual DRAM traffic, C5 scaling or commercial advantage.
Available packets contain no future observations. Weights, decoder and public
information are unchanged. Remote CPU qualification remains a separate CI output.

[Batch audit](batch-audit.json), [rejected per-record cache](cache-audit.json),
[optimized artifact SHA256](evidence-index.json). Extract `batched-agent.tar.gz`
and run `python tac/runtime.py < tac/demo-requests.jsonl` with CPU PyTorch.
Original first-seed export remains intact. This is learned control on the bounded
registered two-entity schema, not arbitrary entity discovery or symbolic programs.
