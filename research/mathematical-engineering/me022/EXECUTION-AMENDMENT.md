# Recorded execution failure and repair

The first attempt used frozen training implementation `f88abac` and execution
head `77f4b78`. The official source passed its hashes; all300updates of original
seed14001 completed and were saved with raw optimizer, sampler and Torch RNG.
The following **post-fit latency diagnostic** raised under Torch2.14.1+cpu:

```
RuntimeError: Inference tensors cannot be saved for backward.
```

The failing call used `torch.inference_mode()` for the inherited TorchScript
module after training. Change only this diagnostic context to `torch.no_grad()`.
Add explicit restore of the completed300-step checkpoint and cached authentic
data, verifying original snapshot SHA before reading it. No original fit is
repeated, no hyperparameter, cohort, update or model equation changes, and no
best-checkpoint selection occurs. All original checkpoint/trace bytes remain.
The first observed original DEV Brier was0.20467005834469937, matching the
published historical seed14001 value to rounding. Other factorial arms had not
started. The repair therefore responds to a runtime failure, not model scores.

The manifest retains the original fitting-source head, adds the execution
amendment and records each arm's execution head. Source building took218.02wall
seconds on the initial attempt. Total wall time of that interrupted attempt was
not instrumented at exit; report resumed wall time separately, never claim it
is complete end-to-end cost. Saved training CPU accounting for the completed
fit is retained. The experiment remains historical-development diagnostics.
