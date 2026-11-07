# Replay this review

The base64 ZIP is a selected, source-pinned snapshot, not an installed model or a full repository clone. It contains no trained weights, private data or credentials. Extract into a new directory; do not overwrite an existing checkout.

```bash
base64 --decode REPLAY_BUNDLE.zip.b64 > REPLAY_BUNDLE.zip
python -m zipfile -e REPLAY_BUNDLE.zip fresh-review
cd fresh-review
python -m venv .venv
. .venv/bin/activate
python -m pip install numpy==2.3.5 pytest==9.2.0
python -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
python verify_bundle.py
python scripts/phone_lab.py plan research/phone_queue/math_engineering_review_20261107.json
mkdir fresh-results
PYTHONPATH=src:. OMP_NUM_THREADS=1 python -m pytest -q tests
PYTHONPATH=src:. python scripts/audit_csl_scan_cancellation.py --output fresh-results/ORIGINAL_REPLAY.json
PYTHONPATH=src:. python scripts/review_math_engineering_20261107.py fresh-results
PYTHONPATH=src:. python scripts/review_vjp_20261107.py fresh-results/VJP_RESULTS.json
PYTHONPATH=src:. python scripts/review_full_storage_20261107.py fresh-results/FULL_STORAGE.json
```

The last command allocates the untrained default 123.8M-parameter model in CPU memory. No optimizer or GPU is used. Allow sufficient RAM for weights and transient allocations; this is not a memory-constrained phone runtime.

The scripts deliberately preserve numerical failures in JSON while completing the audit successfully. Passing pytest means the checks and recorded counterexamples behave as expected, not that the numerical gate is green. Review `failed_rows`, per-argument errors, and source hashes. Runtime/library/platform differences can change rounding; do not demand bitwise equality of all receipts or erase new discrepancies.

`script_initial.py` records the initial coefficient-label mistake. `script_phase2.py` is the exact script behind `RESULTS_PHASE2.json`. For historical execution, copy the desired snapshot into `scripts/` under a new name before running it, so its repository-root calculation remains correct. Keep separate output directories.

The selected PNDS Lean source files are included for source review and hash verification only. This archive does not contain the Lean/Mathlib build environment. The historical successful build and explicit axiom evidence are linked in REVIEW.md and summarized in LEAN_EVIDENCE.json; new Lean compilation is not implied.

No full repository suite or fresh hosted CI has been executed by this review. The branch is an internal draft. Production model sources and historical result files are unchanged.
