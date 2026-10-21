# ME043 / ME043B original model and evidence delivery

All160 original downstream models, complete optimizer/RNG states, inference
state dictionaries, real historical-development forecasts/truths/router arrays,
loss traces and source provenance are retained in repository-owned gzip tar
archives. Eight original A parent fits are included and reused by B. No model
was reconstructed by retraining for delivery.

- [A archive directory and SHA256 manifest](../../model-artifacts/me043/)
- [B archive directory and SHA256 manifest](../../model-artifacts/me043b/)
- [Reproduction protocol](PROTOCOL.md) and [post-A B protocol](FOLLOWUP-B-PROTOCOL.md)

Each directory has four bounded domain/lane archives. INDEX.json inside each
archive gives SHA256 and bytes for every payload. Parent models and common
receipts appear in both lanes; duplicate payloads must agree exactly. The
manifests bind original training source commits. Public publisher files are
checksum-pinned and fetched for exact historical-development replay; the archive
contains no invented or reconstructed outcome labels.

Reproduction in the pinned environment (Torch2.14.1+cpu, NumPy2.3.5, Python3.12):

```bash
python scripts/me043_artifacts.py verify research/model-artifacts/me043 restored-a
python scripts/me043_artifacts.py verify research/model-artifacts/me043b restored-b
python scripts/me043_artifacts.py metrics restored-a
python scripts/me043_artifacts.py metrics restored-b
python -m scripts.me043_train_adaptive --out restored-a --replay
python -m scripts.me043b_train_scheduled --out restored-b --replay
```

Do not run either training command without --replay when restoring models.
Set OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS=1 for the original CPU
convention. Replay is exact in the original environment; cross-hardware behavior
is separately verified by CI rather than assumed. Scheduled-model resume must
restore optimizer/RNG/sampler and set model._training_update to saved updates,
so the every-eighth-step schedule continues from the correct phase.

Optimizer audit checks every tensor's update clock, including floor(updates/8)
for B native/router/gain coefficients. Forecast replay is bit-exact. Float64
metric reduction uses a derived operation-count bound; its initial strict
reduction-equality check failed and its original logs remain. This does not
relax forecast replay or hide the older ME035B precision exception.

The branch workflow restores these actual archives, checks all optimizer states,
replays every original model and offers an additional90-day Actions download.
Only actual completed CI receipts establish whether the remote check passed.
Repository archives remain available after temporary Actions artifacts expire.
No release upload, merge, paid compute or deployment is performed.
