# ME027 — state-value gate fails; learning curves expose a readout/training bottleneck

All twelve registered real-data fits completed. The **primary engineering gate
failed**: replacing TAC's post-RMS learned-state features with their FIT mean
improves Brier in seed 14001 (−0.000159266), while worsening it in 14002 and
14003 (+0.000978836, +0.001102840). Required: at least +0.0002 in every seed.
This is an OOD reliance diagnostic, not causal identification or proof that
native state contains no useful information.

The primary 20 charged CPU-second TAC−MLP mean Brier is −0.007219302, favorable
in every seed. Mean ECE is .013956617 better. All twelve 20s spends fall within
20.000041–20.129685 seconds, satisfying the registered 1% tolerance. However,
the MLP performs thousands of updates and overfits relative to earlier fixed
checkpoints. Its CPU score does not establish useful native memory. Do not
promote TAC, the common-summary interface or the previously failed kernel head.

## Separate resource views

| Model | 50 updates Brier | 300 updates Brier | 300-update mean CPU s | 20s CPU Brier | Mean updates at 20s |
|---|---:|---:|---:|---:|---:|
| tac | 0.186103 | 0.166406 | 20.906 | 0.167642 | 287.3 |
| gru | 0.186305 | 0.166915 | 15.232 | 0.166125 | 395.3 |
| transformer | 0.186522 | 0.167091 | 55.360 | 0.177683 | 109.0 |
| mlp | 0.164508 | 0.165386 | 1.746 | 0.174862 | 3477.3 |

The feature MLP learns much faster per update: at 50 steps its mean Brier is
.164508 versus native .186103. At 300 steps it remains better on average,
.165386 versus .166406, though not in every seed. Under the forced 20s budget
it reaches roughly 3477 updates and mean Brier worsens to .174862. These are
registered points, not retrospectively selected best checkpoints. All curves,
including unfavorable points, remain in table.md and the machine record.

The GRU's feature replacement worsens Brier in all three primary checkpoints
(+.000494567, +.001963184, +.000877903); Transformer's is mixed. These remain
secondary OOD diagnostics. No trained architecture superiority is asserted.

## Why this is not a fully matched architecture win

Trainable counts are 135067 native, 135125 GRU, 134907 Transformer and 135057
MLP. All include 128000 embedding parameters and active scored paths; no inert
padding. All receive identical source channels, sample-stream prefixes, four
causal targets and the same 26-feature construction. Learned embedding values
can differ after fitting. The MLP allocates capacity to a nonlinear common
feature head; sequence models use additive linear state/common heads. Thus
state presence, representation and readout/optimization are not isolated.

Parameter and feature/tuning contracts pass. The fixed-step lane matches data
presentations but actual compute differs; the CPU lane matches charged time
but data exposure differs. **Simultaneous resource parity FAILS.** Full FLOP,
energy and whole-study-cost equivalence remain unestablished. Native/GRU/MLP
are not declared to beat Transformers. Preferred ME024 RMS remains unchanged.

A user-cluster bootstrap over 441 historical DEV users gives primary paired
loss interval [−.009272004, −.005193193], conditional on the three frozen seeds.
It does not correct reused-development bias, unequal exposure or overfitting
and is not a model-selection or independent-generalization certificate.

## Mathematical and execution evidence

Native equations/initial weights match the established core, and standard GRU
cell/sequence equivalence is checked. Nonzero-head prefix causality, an unmasked
attention falsifier, common features, active gradients, serialization and
mean-feature logit preservation pass. The mathematical note separates Bayes
conditional predictive information from finite-model approximation/optimization
error; the latter prevents interpreting score differences as intrinsic state
information in this experiment.

All 72 budget checkpoints replay 428,544 predictions exactly; twelve primary
standalone exports replay 71,424 probabilities exactly. All optimizer states,
steps, sampled exposures and generator draws reconstruct. CPU first-crossing
logic is preserved in frozen source, but there is no independent per-step
clock log to reconstruct timing truth. Cumulative charges and setup conventions
are explicit. All inference latency samples and bytes are retained; process
RSS is not isolated training/inference memory.

Profiler passes on eight real FIT windows, seed 14001/step300, perform forward
and backward without optimizer steps. Matrix FLOP subtotals are lower bounds;
scalar add/mul arguments and opaque kernels are not fully counted. They cannot
establish a full compute match. No fit is repeated during profiling or replay.

The fitting source/architecture/contract were pushed at
`119acd36e2ebb786e49b39a378b34bc632061b7a` before fitting. Later mathematical
interpretation notes and audit/reporting files do not alter any frozen fit,
selection rule or threshold. Actual whole study: 484.437 process CPU seconds, 484.494 wall seconds.
Only the authenticated 10,776 FIT and 5,952 already-exposed DEV windows are used.
No reserved cohorts, synthetic capability data or new test outcomes were opened.
All negatives, raw weights/Adam/RNG, probabilities, traces and exports remain.

See REPRODUCE.md and ARTIFACT.md. ARCHITECTURE-DIRECTION.md derives a small native
nonlinear-observable head as a future hypothesis; it has not been fitted here.
Nothing is merged, deployed or commercially qualified.
