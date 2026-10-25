# ME093 original artifacts and restoration contract

At this pre-result update, five original fits are executing. There are no claimed
ME093 scores or completed artifacts yet. This file describes required delivery;
final receipts and original forecasts must be checked before marking complete.

* Registration FAIR_COMPARISON.json was committed51120067 before first fit.
* Seeds93001–93005: each starts from its fixed indexed ME092original92001–92005,
  retains92,017meaningful values, adds4,578local-noise values and resets AdamW.
  Weight-only warm initialization is not original-optimizer continuation.
* Each final.pt must include complete weights, all optimizer groups/moments,
  torch/NumPy/Python RNG, fixed schedule, FIT-only buffers, parameter-budget
  receipt binding, actual update count and explicit parent hashes/lineage.
* Every final.pt, live-budget.json, FIT_RECEIPT.json and original fit log must
  be repository-owned, plus ALL5 FIT_INDEX.json, **before any REPORT scoring**.
* Same licensed numeric source/cache and exact Git-owned overhead pixels asME091.
  SOURCE_QUALIFICATION.json pins source roles/revision/checksums. Run
  `python scripts/me091_restore_pixels.py` in a clean checkout before scoring;
  do not re-decode videos with different codec/resize defaults.
* Run `PYTHONPATH=src:. python scripts/me093_train_world.py finalize --study me093`,
  then final parity validation. Original frozen receipt check rejects absent,
  changed or not-yet-Git-owned model bytes.
* Original score mode writes per-seed ORIGINAL_PREDICTIONS.json and
  ORIGINAL_PREDICTIONS_H32.json exactly once, plus ORIGINAL_SCORE_INDEX.json.
  Report script applies every unchanged joint gate; no individual-seed selection.
* Clean independent CI runs restore + replay (no fitting), full field audit and
  optional compiled/physical boundary restoration under rtol1e-5/atol1e-6.
  Original score denominators are expected27,600model-forecast records across
  five seeds and8/32history conditions, not27,600independent real episodes.
* Pipeline13,312updates; retained native/visual own-lineage9,216each,
  dynamic-location5,120,newlocal-noise3,072. Repeated exposure is not new data.

For normalized fixed-profile executables run scripts/me093_compile.py; for actual
raw qpos/reference/pixel→physical-quantile/state graph run
scripts/me093_physical_boundary.py. Programs are derived from the first original,
not extra learned models. Source-camera decode, authority, network and handler
hashing are outside their timing. All originals, failures and numerical exceptions
remain immutable; compiled paths cannot silently replace original replay.

Completion: all5originals frozen5ec26e40, final live counts/optimizer audit frozen
e5a2f36b before REPORT. Actual35,316,679dense supervised target presentations across
five fits; these are repeated real observations, not unique independent data.
Numerical full gateFAIL: quietH1 and additional99.58333%CI. No further ME093fits.
Independent original score replay remains pending at this update.
