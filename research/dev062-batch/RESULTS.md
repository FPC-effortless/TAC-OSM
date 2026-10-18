# Available-packet inference qualification

The unchanged trained TAC92601 policy qualifies locally for **batched intake**.
Mean model CPU on the registered first-seed traces: base 18.679ms,
batched 9.717ms,
batched cached 8.864ms.
Batch/base ratio 0.5202; cache/equally-batched ratio
0.9123. The additional cache gain misses the
registered10% threshold. The original unbatched cache was slower and stays rejected.

Each variant reconstructed23 checkpoints and6784 recorded trajectories with exact
choices/costs/outcomes under strict1e-8 checks. Maximum numerical differences are
0 for batch and2.85e-14 for batch+cache. These recordings share fixed worlds;
they are not independent physical trials. Remote qualification remains UNKNOWN.

The timing covers all model observation/decision work including terminal receipts,
but excludes JSON/tensor creation, simulation and verification. It is a narrow CPU
measurement, not production latency or measured DRAM traffic. No learned weights,
posterior update, physical action rule or termination override was added. Batching
uses only records already available before each decision. Full-bank history reads
remain bounded linear work; this does not establish C5 indexed scaling.

[Protocol](PROTOCOL.md), [complete measurements](audit.json),
[portable artifact and SHA256](export-index.json). Extract `batched-agent.tar.gz`;
run `python tac/runtime.py < tac/demo-requests.jsonl`. CPU PyTorch is the sole
non-standard runtime dependency. The original export remains unchanged.
