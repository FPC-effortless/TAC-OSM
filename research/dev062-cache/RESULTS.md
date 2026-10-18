# Incremental cache — rejected engineering candidate

The cached learned-state runtime preserved all checked actions and outcomes but
used1.508x the original total model CPU on the registered first-seed traces.
It fails the registered efficiency gate and is not the preferred export.

23 checkpoints,6784 recorded trajectories and3415662 telemetry values were checked;
maximum local numerical difference2.85e-14. Correctness does not imply efficiency.
Per-record dispatch offsets avoided recurrent recomputation in this workload.
[Protocol](PROTOCOL.md), [complete measurements](audit.json).
