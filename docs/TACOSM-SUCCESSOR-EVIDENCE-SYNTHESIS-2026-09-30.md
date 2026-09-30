# TAC-OSM successor evidence synthesis — 2026-09-30

This document consolidates the completed successor-architecture diagnostics.
It does not replace their individual preregistrations or results.

## Evidence ladder

| Experiment | Question | Result | Interpretation boundary |
|---|---|---|---|
| REP-001 | Can the existing successor router select among collapsed candidate observations? | A=1.000; learned≈no-learning≈0.18; exact execution=1.000 | Selection failure; candidate representation was insufficient for the tested topology distinction. |
| REP-002 | Does explicit candidate executable topology remove the information collision? | 8/8 distinct topology observations; analytic=1.000; learned=0.2219; no-learning=0.1266 | Explicit topology makes the tested relation identifiable and partially learnable. |
| REP-003 | Can the learner use topology for a semantic dependency, rather than exact edge-mask equality? | learned=0.2008; no-learning=0.0977; analytic=1.000 | Some semantic structural signal is learned, but performance is seed-unstable. |
| REP-004 | Is simple training-time exploration the main learner bottleneck? | learned exploration=0.2094 vs baseline=0.2008 pooled | The registered epsilon-greedy intervention is insufficient as the sole repair. |
| REP-005 | Can the semantic requirement cross a causal persistent-state boundary? | learned=0.1633; no-learning=0.1023; analytic=1.000; reset=160/160 fail-closed | Persistent state is usable in the tested one-boundary synthetic loop; long-horizon memory is not established. |
| REP-006 | Can a relevant state item be identified from semantic content when its address is opaque? | **pending** | Current registered state-addressing experiment; no result until the passing CI run completes. |

## Architecture decomposition

The completed sequence now separates five distinct boundaries:

1. candidate observability;
2. semantic structural representability;
3. training dynamics;
4. temporal state transport;
5. semantic state addressing.

The executor is not the current limiting mechanism in these diagnostics:
exact execution remained 1.000 in the registered learned/no-learning program
selection arms where execution was evaluated.

## Current compute model

### Candidate routing

REP-003/REP-005 use the graph-program router:

`C_route = 8*5 + H*(16*7 + 8*16 + 8)` MACs.

At H=8, this is 2,024 MACs per routing decision before nonlinear functions and
software overhead. The term is linear in candidate population H.

### State addressing

REP-006 uses the semantic state addressor:

`C_state_address = 8*5 + M*(8*5 + 8)` MACs.

At M=8, this is 424 MACs before dictionary reads, state materialization and
software overhead. The registered implementation scans the state pool, so
this cost is linear in M.

### Execution

The exact synthetic executor operates on a fixed seven-edge substrate and the
registered programs activate two edges. Its work is therefore effectively
constant with respect to H in these experiments.

## What the architecture does NOT yet demonstrate

- sublinear semantic state addressing;
- sublinear candidate retrieval;
- computation proportional to a measured relevant subset R;
- capability/computation parity curves;
- large-history scaling with semantic state addressing;
- natural-language semantic program selection;
- long-horizon persistent learning.

The repository's C5 claim remains UNTESTED.

## Next measurement boundary

REP-006 is the current gate. A passing result should be followed by a scaled
state-pool experiment that varies M while holding semantic task difficulty
fixed, followed by a selective-retrieval experiment that measures:

`C_address(M) + C_route(K) + C_execute(|R|) + C_verify`

against the conventional:

`C_full(H)`.

The capability parity condition must be declared before that confirmatory
measurement. Until then, an O(H) scan with a small constant factor is still
linear routing, not selective computation.

## Provenance

Completed result records:

- TACOSM-SUCCESSOR-REP-001
- TACOSM-IDENTIFIABILITY-REP-002
- TACOSM-SEMANTIC-PROGRAM-REP-003
- TACOSM-LEARN-REP-004
- TACOSM-PERSISTENT-SEMANTIC-REP-005

Current registered experiment:

- TACOSM-STATE-REP-006