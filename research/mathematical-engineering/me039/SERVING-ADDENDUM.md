# Post hoc comparison with the existing serialized serving baseline

Added AFTER observing the original eager-versus-scripted-core timing results,
BEFORE measuring the paired comparison below. This is an explicitly post hoc
engineering extension; it cannot replace the failed primary training targets.
ME038 already exports complete traced physical-input models. A speedup over its
eager Python path must not be presented as a speedup over that deployed format.

Compare original ME038 physical TorchScript exports with ME039's newly exported
scripted-core physical models on the same eight originally registered native
cases. Load immutable original bytes, verify source hashes and both original
physical-input probes exactly. On B1, five warmups/path, thirty timed samples/path,
alternate path order each trial; single thread, no other local CPU-heavy work.
Report per-case and geometric-mean old/new ratios, all cases, and serialized file
bytes for all sixty models. No selection, new fit, accuracy score, new claim gate
or change to prior targets. Preserve negative/neutral serving results. If the
new export is slower, retain the original as the recommended serving format.
