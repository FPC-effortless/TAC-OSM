# LANGMATH098 state separation diagnostic

This is an **untrained mechanism test**, not a learned-intelligence experiment.
It tests the unchanged ContextualAssociativeTAC098 source from PR #668 at
427db08e1d2c971b4fb4c3c579214e289211762e. Git blob and SHA256 are checked before
every run. No old 086 checkpoint is loaded as 098. No optimizer or TEST access.

The controlled question is whether current input, carried information, and the
final predictive head can be separated mechanically. It does not test whether
the model has learned which information is useful.

## Executed local result

Three random initializations: 98001, 98002, 98003. **54 checks PASS, 0 FAIL**
after the documented duplicate-batch FP32 test correction. A fresh process
reloaded all three original initializations and acquired states and reproduced
the saved logits exactly. Initial failed receipt is retained unchanged.

| Condition | Observation | Interpretation |
|---|---|---|
| A: only the same current byte, fresh state | Same predictions for conflicting histories | A balanced two-label hidden-history task has a 50% information ceiling; no learned score is claimed |
| C: carry all native state | Earlier byte changes the later logits in all three initializations | History affects the computation, without demonstrating accurate recall |
| B: export/load all state | Bit-identical predictions and updated state | State storage can be external to the model instance; the system remains stateful |
| B: replay the same raw bytes from zero | Matches carried inference within registered FP32 tolerance | History can be supplied externally, with extra processing cost |
| Perturb final predictive head | Predictions change; every updated state tensor stays exactly the same | The final readout is functionally separable from the forward state updater |
| Disable associative-bank writes | Banks stay fixed; context, recent bytes and previous keys change | A no-write switch is not a stateless condition |
| Reset only associative banks | Predictions change, but other history remains | Bank reset is not a valid full-state-removal control |
| Future poison / entity isolation / detached acquisition | Specified checks PASS | These specific information/graph boundaries held; no universal leakage certification |
| D: state defined as future predictions | Not implemented or trained | No predictive-state learning or sufficiency result |

The model has **100,370** learned parameters and **432 FP32 state values =
1,728 bytes per stream**, excluding owners/clock and serialization overhead.
The terminal LayerNorm/FFN head contains **80,896** parameters and takes 162
features. The remaining **19,474** include shared feature and state machinery.
Separating this final head does not establish that the entire predictive
competence is independent of state formation; the feature builder still uses
history and shared weights.

## Measured local costs

AMD EPYC 9V74, PyTorch 2.5.1 CPU FP32, one thread, batch one, five warmups and
30 samples. These are constructed untrained measurements. Local p50:

| History bytes | Native carried step (ms) | Load external snapshot + step (ms) | Raw-history replay + query (ms) |
|---:|---:|---:|---:|
| 64 | 0.336 | 0.735 | 1.098 |
| 256 | 0.340 | 0.743 | 2.044 |
| 1024 | 0.348 | 0.706 | 5.977 |

The current-only original model takes about 0.35–0.37 ms on this fixture and
executes the same graph with freshly allocated zero state. It does not
physically remove memory modules, so resetting state does not save their
arithmetic. This is not a trained quality/cost comparison or production latency.

The PyTorch profiler reports **183,636 supported-operator FLOPs** per byte
step, with or without carried state. Replay at 65, 257 and 1,025 bytes reports
11,925,680; 47,153,072; and 188,062,640 respectively. The receipt lists every
counted and uncounted operator. These are **partial ATen FLOP counts**, not
complete model/hardware FLOPs. Serialization time is included only in the
snapshot-load condition; network transport and snapshot-writing are excluded.
Raw history storage, Python container overhead and tensor payload are separate.

## Reproduction

Standalone evidence package:

```bash
python scripts/validate_research_parity.py --manifest FAIR_COMPARISON.json --phase preflight
python run_diagnostic.py --output results/NEW_DIAGNOSTIC.json
python replay.py --source-model source/model.py --results results --output results/NEW_REPLAY.json
python scripts/validate_research_parity.py --manifest FAIR_COMPARISON.json --phase final
```

In the full repository, pass `--source-model
research/mathematical-engineering/langmath098/model.py`. Use new output paths;
the runner refuses to overwrite a diagnostic JSON. Source model weights are
saved under INITIAL_AND_REPLAY filenames; these are untrained snapshots.
The GitHub workflow reruns the source's original NumPy/gradient/probe tests,
these diagnostics, and fresh-process replay on its own runner.

## Remaining scientific experiment

A learned four-way study still needs preregistered real FIT/DEV data, exact
capacity and training-budget checks, frozen splits and source receipts. A must
be trained on its permitted current observations rather than judged only after
ablating a recurrent model. B needs an actual external-state policy and
end-to-end retrieval/replay accounting. C is the exact 098 native model. D
requires a separately registered prediction-conditioned state updater, targets
defined from legally observed futures, and held-out action/query tests of
sufficiency. No oracle sufficient state may be counted as discovered memory.

Measure prediction NLL/accuracy, full reset and same-clock donor effects,
hidden-history utility, unseen-source transfer, latency, state size and actual
action outcomes where the source provides them. Do not manufacture closed-loop
outcomes from offline text or call them real-world decision success.

The repository forbids a standalone synthetic-accuracy training campaign:
`research/REAL-WORLD-EVIDENCE-RULES.md`, rule 2. This diagnostic therefore spends
zero optimizer steps. H1 (prediction produces state), H2 (state enables useful
intelligence) and H3 (coupled learning) remain unresolved by these measurements.
