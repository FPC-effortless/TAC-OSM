# ME044 original models and research evidence

[Repository-owned model archive directory](../../model-artifacts/me044/).
Eight bounded standard-library tar.xz shards retain all80 original downstream
checkpoint files, optimizer and RNG states, inference state dictionaries, loss
and CPU traces, full original historical-DEV forecast/truth/trust/cheap arrays,
source provenance and eight original AR parent fits. No delivery model is
recreated through training. Optional scripted native branch exports are also
preserved when construction succeeds.

Each archive has INDEX.json with SHA256 and bytes for every payload. The outer
me044-manifest.json pins each archive and the original training source commit.
XZ uses a64MiB dictionary to losslessly compress repeated records; no learned
weight quantization or runtime-memory reduction is claimed. Each shard must
remain below49MB. Publisher source files are retrieved by pinned original hash
for replay; no missing outcomes are generated.

Original pinned environment: Python3.12, Torch2.14.1+cpu, NumPy2.3.5. Install the
checkout with `pip install -e .` and set OMP_NUM_THREADS, MKL_NUM_THREADS and
OPENBLAS_NUM_THREADS to1. Restore and audit without fitting:

```bash
python scripts/me044_artifacts.py verify research/model-artifacts/me044 restored-me044
python -m scripts.me044_train_multiscale --out restored-me044 --audit
```

The audit verifies every optimizer's scheduled step clock, all saved forecasts
bit exactly, actual original truth/trust/cheap arrays, and independently recomputed
real MSE/MAE. It also compares all24 fixed-update cheap state dictionaries to
the original context model's parameters. No new final-test period is opened.
A training resume must use saved updates to continue the every-fourth or prior
every-eighth expert schedule, restore optimizer and both RNG states, and retain
the exact source and parameter-group/clipping conventions.

Full local replay and separately observed remote CI receipts establish their
own results; cross-hardware equality is not assumed. The workflow restores the
actual pushed archive bytes and provides an additional90-day Actions artifact.
Repository copies remain after temporary downloads expire. Original ME043
strict metric-reduction equality failure and ME035B precision exception are
unchanged; forecast replay tolerances have not been relaxed.

No merge, promotion, paid compute, release asset upload or deployment.

For a native model, optional compiled serving uses its original full/inference
checkpoint together with the sibling scripted-native-branch.pt. Build the model
with its recorded arm/horizon/seed/period, load the state dictionary, then replace
model.branch with torch.jit.load of that sibling file and call model.eval().
The archive's inference.json records every compiled-export SHA256 and its paired
full-model timing. To verify the actual saved exports without refitting or timing:

```bash
python -m scripts.me044_inference --out restored-me044 --verify-only
```
