# Executor interruption and recovery

After14complete fresh groups the executor restarted. Groups96815/96816 had only
partial progress curves and no model checkpoints or validation results. Preserve
those directories under `interrupted-executor`; restart the SAME seeds using the
SAME frozen `group()` function, source, data hashes, LR selection, all four arms
and update counts. No performance-based retry or seed exclusion is permitted.

The fitting commit remains a6d8cf3 while fitting functions capture it. Hash the
recovery wrapper and locked selection into `resumption.json` BEFORE any restarted
worker. No test target may open before the original full-matrix validation rule.

Exact CPU usage of lost processes is unknown: conservatively charge the full
1,800CPU-second seed-group cap for each interrupted process. Bound setup using
twice its persisted wall interval plus10startup seconds, reflecting twoCPUquota;
report that as a conservative charge, not a measurement. Keep the registered
24,000CPU-second global cap and1,800CPU-second per-group cap unchanged.

Administrative scheduling deviation, recorded before resumed fitting: permit
single-group admission if the remaining budget cannot reserve two full caps.
This changes neither training nor information/quality gates. If insufficient
budget remains, close INCOMPLETE without test opening or promotion. Preserve
this deviation, the lost-accounting limitation and all partial curves in release.
