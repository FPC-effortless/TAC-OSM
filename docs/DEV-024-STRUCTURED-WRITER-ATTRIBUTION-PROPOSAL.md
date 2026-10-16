# DEV-024 proposal: does the structured writer need to learn?

**Proposal only; no implementation, execution or findings.** Sampling,
thresholds and code must be fixed before the next held-out outcomes are opened.

DEV-023's masked-write/read plus local-aging variant passed both H512 development
gates across five seeds and two regimes. Coordinate isolation alone improved
means but failed reliability; local aging with unmasked writes/reads stayed near
chance. This supports a useful structured architecture for the tested task.
It does not show that training the writer, rather than encoder/readout learning
on top of its fixed dynamics, was responsible.

## Matched learning interventions

Retain the successful masked-local computation and compare four controls from
identical initial tensors and public histories:

| Arm | Event encoder | Memory writer | Query encoder/readout/head |
|---|---|---|---|
| Joint | Train | Train | Train |
| Frozen writer | Train | Frozen at initialization | Train |
| Frozen encoder | Frozen at initialization | Train | Train |
| Both frozen | Frozen at initialization | Frozen at initialization | Train |

Use learned and frozen-recurrence schema41 references. Five fresh seeds × two
regimes × six arms would preserve the recent 60-cell budget. Freeze new numerical
namespaces and optimizer/episode/receipt exposures prospectively. Reusing the
writer-training controls from DEV-018 under this changed graph is a new test,
not a retroactive reinterpretation of DEV-018's failed native writer attribution.

The primary writer-learning contrast should compare joint against frozen writer.
The fixed-encoder writer contrast supplies a second attribution check. Require
component hashes/gradient traces to prove that only registered components
changed; frozen means *random initialization*, never a previously trained writer.
Active training parameter counts necessarily differ and must be reported.

Keep pre-sensing H512 accuracy, reset/swap dependence, held-out H1024 strata,
per-seed failures, unnecessary services and full physical costs. Predetermine
the accuracy-gain margin, paired-seed interval and reliability rules before
training. Preserve the existing selected-versus-model-only economic constraint;
do not tune it to rescue previous small failures.

## Interpretation and remaining gaps

If a fixed writer matches joint performance, the justified finding is useful
structural memory plus encoder/readout learning, with learned writer value still
unproven. If both writer-learning contrasts pass, that supports a learned-writer
contribution within this specific public-slot architecture, not full native TAC
or CASM success. Failed contrasts do not prove learning never helps.

Write-versus-read coordinate masking remains bundled and merits a separate
factorial if learning attribution warrants it. Variable entity counts require a
new validated public input/query/world boundary; another fixed-eight experiment
cannot license such transfer claims. These limitations survive any positive
DEV-024 development result.
