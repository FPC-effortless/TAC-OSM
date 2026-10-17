# DEV-052B evaluation isolation correction

Execution v1 was prospectively frozen in 5b79d0c and implemented at ee9b243.
Its smoke path and unit tests accidentally used its reserved final evaluation
namespace (5521004/5521005). The issue was found by reviewing seed selection,
before inspecting numerical full-matrix results. No architecture or recipe
was selected from evaluation outcomes.

The original full five-seed matrix was allowed to finish and is preserved under
execution-v1-excluded/. Its results are excluded from the scientific endpoint.
Its earlier smoke and separate smoke-audit outputs are also preserved.

The correction was registered at b7a82cf under execution ID DEV-052B-FRESH1:
new final evaluation seeds 5522004/5522005, smoke seeds offset by 10000000,
and unit tests moved to a separate 852100x namespace. A regression gate asserts
these namespaces are disjoint. Training data, all architectures, 256-update
budgets, optimizer, calibration procedure, and cost/failure acceptance rules
were unchanged. The new protocol was committed before the fresh run began.

This is an integrity correction and a fresh development evaluation, not an
independent confirmatory replication. Both complete runs and all smoke fits
must be counted in research compute; neither bad seeds nor unsuccessful arms
may be omitted from the archive.
