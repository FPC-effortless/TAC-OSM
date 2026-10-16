# DEV-021 proposal: frozen suffix transport before architectural redesign

**Status: proposed diagnostic protocol; no DEV-021 execution or results.**
This follow-up is conditional on DEV-020 not establishing a reliable state-probe
versus read-probe distinction. It does not modify DEV-018 or DEV-020, train a new
world model, or establish a new architectural success gate.

## Question and algebra

Do the *trained* TAC decay and erase operations attenuate a known historical
distinction during an identical public suffix, and can their contributions be
separated without relying on a failed classifier?

For one expert, write the actual `DeltaMemory.update` as

```text
S_next = d*S + a*(-(S @ (e*k)) outer k + wv outer k)
```

Here `d` is scalar decay, `a` is assignment, `e` is key-side erase,
`k` is the normalized write key, and `wv` is the value-side write vector.
All coefficients depend on the current public input and frozen parameters,
not on `S`. Once paired streams have identical public inputs, their state
difference therefore evolves exactly as

```text
D_next = d*D - a*(D @ (e*k)) outer k
```

The additive write cancels. This identity allows an exact frozen-model
decomposition. It does not imply that the entire difference encodes the
desired causal label, or that the resulting predictions are correct.

## Evidence and sampling fixed before execution

- Reload the five DEV-020 arm checkpoints from DEV-018 run 38802569971;
  verify report hashes and parameter hashes. Analyze the recurrence equation
  only for `tac_joint` and `tac_encoder`; retain schema41, GRU and memoryless
  as observational distance controls.
- Use all three seeds and both training regimes without selection.
- Generate fresh `dev021-transport` histories, never the DEV-020 test domain:
  H512, pending/reissued cancellation, at/after deadlines, delays 32/128/512,
  16 repetitions per cell, giving 384 episodes / 192 pairs per seed-regime.
- Locate each pair's last unequal public observation from the actual stream.
  Verify equality of **all** public encoder inputs after that point. Never
  assume equality solely from a shared random seed or final observation.
- Record full states immediately after the last unequal input, at H32/H128/H512,
  and the final native read features and logits. Use identical batches for
  paired traces. Record both L2 and relative attenuation from the divergence
  boundary, with zero initial differences flagged separately.

## Decomposition and interventions

1. Replay the unmodified recurrence and independently transport its paired
   state difference using the equation above. Require numerical agreement
   throughout the suffix, with the runtime, dtype and tolerance recorded
   before measurement. A failure invalidates the decomposition.
2. Record actual per-expert decay products (sum logs for stability), erase
   contributions and directional attenuation. Do not substitute initialized
   `sigmoid(5)` for trained values. Ratios of norms are descriptive and need
   not factor into independent causal contributions.
3. Transport the same initial difference with decay set to one, with erase
   set to zero, and with both disabled. Change only the named term; keep the
   actual encoded suffix, keys, assignment and frozen parameters identical.
   Report the full 2x2 factorial grid rather than choosing a favorable variant.
   These are interventions on *difference dynamics*, not trained predictors.
4. If a later experiment runs altered full state trajectories through the
   native reader, preregister it separately. Removing a contraction can grow
   noise and shift state distributions; greater separation alone does not
   demonstrate recovered memory or improved maintenance decisions.

## Probe limitations to resolve separately

DEV-020 uses only 288 probe-training episodes, a 6,158-feature dense-state
interface and a training standard-deviation floor of 0.05. Weak signals can
remain weak after that normalization. Failed probes cannot distinguish absent
information from insufficient capacity, conditioning, sample size or training.
Any new probe comparison must use fresh training/validation/test domains,
training-only scaling and validation-only selection, then freeze the selected
recipe before opening the new test outcomes. It cannot replace DEV-020's fixed
recipe retrospectively.

## Decision boundary

Selective attenuation under frozen decay/erase interventions would justify a
subsequent controlled retention intervention, with independently trained models
and physical service-cost evaluation. Preserved state differences alone would
not justify a reader redesign. Neither outcome establishes causal-state
recoverability without task-specific evidence. Prior TAC capability and
memory-attribution gates remain failed until a separately specified experiment
actually passes them.
