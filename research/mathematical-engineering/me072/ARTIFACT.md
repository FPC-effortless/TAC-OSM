# ME072 original world-model loading and real-score replay

Original native checkpoint bytes, optimizer/Torch RNG, source/preprocessing,
schedule SHA, epoch traces and measured exposure are preserved for each seed
under `research/replay-bundles/ME072/`. Training does not score REPORT. All
five final models must be committed before scoring. Models are not substituted
with retrained copies or selected retrospectively.

Dependencies are pinned in requirements.txt. The inherited BPI2012 loader
verifies exact publisher file identity and can fetch a pinned byte-identical
mirror; raw traces remain external under the publisher's 4TU terms. Additional
new clock/survival/mark priors and the inherited FIT-only history table are
stored as exact Git-owned values; replay never refits numerical preprocessing.

At the final pinned commit in a clean checkout:

```bash
pip install numpy==2.3.5 pytest==9.1.1
pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
export PYTHONPATH=src:.
python scripts/validate_research_parity.py --manifest research/replay-bundles/ME072/FINAL_REGISTRATION.json --phase final
python scripts/me072_marked_world.py --phase replay --cache /tmp/me072-source
```

The replay CLI verifies original model/checkpoint bytes against Git, strictly
loads all five models and optimizers, rebuilds available observed prefixes,
and recomputes distribution arrays and factual scores for every supported
query/horizon. Original arrays and score bytes are restored even on failure;
failed candidate outputs and fresh failure receipts are retained. Replay
does not update weights, tune priors, refit calibration or score original TEST.
The scoped GitHub workflow performs the independent clean-checkout audit.

ORIGINAL_FORECASTS.npz has one key per seed/query age. Each array row is one
reached original REPORT case in source order; second axis is horizons1/2/4/8.
Last-axis fields are joint log probability of all activity classes followed by
EOR, log zero-time mass, two shared mixture log weights, two component log-time
locations, two log standard deviations, and three log1p timing quantiles.
Timing parameters/quantiles condition on existence of a future record; there
is no artificial timestamp attached to EOR. Case ordering is checked by ID
digests, while original targets are reacquired from the lawful external source.

REPLAY_MANIFEST.json indexes model, data, configuration, loader, arrays and
score hashes. Environment records and original training output accompany the
checkpoint manifest. A checkpoint/source availability or numerical mismatch
is FAIL/BLOCKED, never waived by a later passing code-only test. Exact training
re-execution or resumability is not claimed; actual original model/evaluation
replay is the required gate.
