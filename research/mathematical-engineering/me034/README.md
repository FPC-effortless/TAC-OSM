# ME034 — Stable-input native recurrence

[Protocol](PROTOCOL.md) · [Mathematics](MATHEMATICS.md) ·
[Verified results](RESULTS.md) · [Artifacts / restoration](ARTIFACT.md)

72completed real-data fits. Freezing R-only pretrained inputs improves native
TAC, but it loses to common-head adaptation on the registered new-user test.
Both multiplicity-adjusted predictive gates fail. The registered stop closes
this KuaiRec native-memory recipe; no architecture is promoted or merged.

User remainder6 is now exposed. Do not run the original fitting/final-access
commands as a new test. Reproduce saved evidence offline:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.me034_audit --out results/me034 --quick
```

Full audit additionally rebuilds raw pipelines, runs fresh Torch-only processes,
profiles trained operators and measures inference. `--verify-only` validates
the frozen/final/export payloads before reading source arrays. Exact environment
versions appear in [environment.json](environment.json). Boosted trees require
scikit-learn1.8.0; they are hashed trusted local research artifacts, not arbitrary
external pickle inputs. The base `tac_osm` package remains Torch-free.
