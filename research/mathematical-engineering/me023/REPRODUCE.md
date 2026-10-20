# Reproduction and replay

Executed environment: Python 3.12, Torch 2.14.1+cpu, NumPy 2.3.5, Linux x 86-64,
AMD EPYC9 V74 CPU and one Torch thread. Matplotlib 3.10.8 generates the curves.
The core is an optional Torch module; importing `tac_osm` alone remains free
of Torch imports and does not change the lightweight interface package.

```bash
python -m pip install -e . 'numpy==2.3.5' 'pytest==9.1.1' 'matplotlib==3.10.8'
python -m pip install 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q
```

Training implementation and complete contract were frozen at `4b17f08` before
any fit. This explicit command executes the 12 registered fits on the already
authenticated parent data. It requires the original ME022 checkpoints and
data snapshot and verifies every original hash before using responses:

```bash
python -m scripts.me023_predictive_research --parent results/me022 --out results/me023-new
```

Every native checkpoint includes raw weights, optimizer, Torch/sampler RNG,
duration statistics, step and contract. The executable TorchScript artifact
also exports `encode`, `readout` and `prefix_logits`, allowing state and causal
prediction inspection without TAC source imports at inference. It expects
original 128-step tensors and applies the exact two-scalar duration transform.

The delivered portable archive excludes raw/derived event records. Restore it
to a fresh directory and replay against the local original parent snapshot:

```bash
mkdir -p results/me023-restored
tar -xzf /path/to/me023-checkpoints-and-records.tar.gz -C results/me023-restored
python -m scripts.me023_audit --root results/me023-restored --parent results/me022
```

When original cached data is unavailable, replay can reconstruct **only the
same early exposed data** from the official publisher archive, without fitting
or opening new cohorts. Exact canonical data fingerprints must match:

```bash
python -m scripts.me023_audit --root results/me023-restored --rebuild-source
python -m scripts.me023_summarize --root results/me023-restored --out results/me023-restored/report
```

The audit verifies all new model/standalone hashes before reading responses.
It replays original, standalone and FIT-mean-state intervention predictions,
checks stored metrics, exact sampler RNG, active capacity, supervision budgets
and common CPU ceilings. Derived singular spectra refer to the bounded learned
readout matrix, not recurrent transition eigenvalues or memory capacity.

Neither replay nor the mathematical CI job trains models or opens reserved
tests. Repeated historical-development inspection cannot establish independent
generalization, causal state utility, decisions or commercial qualification.
