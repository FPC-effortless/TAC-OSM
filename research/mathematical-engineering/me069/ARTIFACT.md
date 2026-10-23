# Original model and scoring replay

The registered ten model/optimizer/Torch-RNG checkpoints are stored directly
under `research/replay-bundles/ME069/checkpoints/`. Their exact SHA-256 identities,
training schedules, consumed exposure and resource receipts are in
`CHECKPOINT_MANIFEST.json` and `FAIR_COMPARISON_FINAL.json` in that bundle.
These are original fitted weights, not a recipe for retraining substitutes.

Use Python 3.12 and the versions in `requirements.txt` (the original local
environment was Python 3.12.14, PyTorch 2.14.1+cpu, NumPy 2.3.5). No accelerator,
paid endpoint, customer credential or private data is needed. Scripts set one
CPU thread. Raw BPI2012 records are not redistributed: download the external
publisher file or pinned byte-identical mirror through the checksum-verifying
loader. The original publisher terms remain applicable; see SOURCE_PROTOCOL.md.

From a clean checkout at the final research commit:

```bash
pip install numpy==2.3.5 pytest==9.1.1
pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
export PYTHONPATH=src:.
python scripts/me069_verify_workflow.py --phase restore --cache /tmp/me069-source
python scripts/me069_verify_workflow.py --phase replay --cache /tmp/me069-source
python scripts/me069_verify_workflow.py --phase summarize --cache /tmp/me069-source
python scripts/validate_research_parity.py --manifest research/replay-bundles/ME069/FAIR_COMPARISON_FINAL.json --phase final
```

`restore` verifies repository checkpoint bytes and strictly loads all original
models and optimizer states before held-out inference. `replay` reloads the
originals and recomputes all DEV and TEST case-score records, comparing with
the Git-owned originals under the preregistered numerical contract. It retains
the original recorded score bytes rather than replacing them with regenerated
scores. This checks actual TEST metrics, not just source code or saved DEV
predictions. Bootstrap summaries condition on the five fitted seeds.

The initial fits used `scripts/me069_train_workflow.py --phase train`; that
phase deliberately does not score held-out cases. Its terminal output and
checkpoint learning traces are preserved with the originals. Do not rerun
training into the same bundle or reinterpret a fresh training run as replay.
The source-qualification commits, subsequent model freeze and independent
restoration receipt preserve the order of access. External downloads becoming
unavailable make full raw-data scoring BLOCKED, even if checkpoint loading
continues to pass.
