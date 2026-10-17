# DEV-050A: public Veritas engineering results

The final amended public generator (`01ab1d1`) completed 640 worlds × six fixed controls across ten seeds. All 3,840 policy episodes were independently reconstructed and re-executed; native traces, public action observations and verifier breakdowns matched exactly. Veritas source was pinned to `b21d829b62d3cab2811e746267ca305e110ed004`.

| Reference policy | Mean cost/entity ± seed SD | Remaining faults | Complete worlds / 640 | Blocked actions |
|---|---:|---:|---:|---:|
| wait | 2.009375 ± 0.184025 | 643 | 158 | 0 |
| service_without_isolation | 3.009375 ± 0.184025 | 643 | 158 | 1280 |
| stale_record | 1.856562 ± 0.127164 | 406 | 297 | 0 |
| service_all_once | 1.706250 ± 0.109052 | 162 | 487 | 0 |
| inspect_once | 1.209062 ± 0.146327 | 162 | 487 | 0 |
| inspect_verify_retry | 0.917578 ± 0.084330 | 0 | 640 | 0 |

Inspection alone did not resolve uncertainty about whether a repair worked: inspect-once left 162 faults. Inspect–verify–retry resolved all 640 worlds at 0.917578 cost/entity versus 1.209063 for inspect-once, counting sensing, isolation, repeat repair and failure penalties. Verification only helps here because an observed unsuccessful first attempt can be followed by a second effective repair.

Service without isolation produced 1,280 blocked actions, and no fault repairs. Blindly servicing every entity once left the same 162 faults as inspect-once, at higher cost because healthy entities were unnecessarily serviced. Confidence and source-authority fields on stale records did not establish truth.

This establishes a working, independently verified transition/evidence boundary for these reference policies. It is not trained TAC-OSM transfer, a canonical Veritas benchmark score, frontier qualification, independent validation of real plant dynamics or a zero-risk deployment certificate. Accurate sensors and guaranteed second-attempt repair are explicit generator assumptions. Native inspection actions are charged by Veritas; the declared TAC sensor adapter supplies the current inspected fault and records that public observation. Hidden state, transition effects and native harness traces are never policy inputs.

Native process, constraints, outcome and evidence dimensions are retained individually in the raw reports. Overall reward must not override a failed requirement. Regression probes confirm that blocked actions receive no service credit, shared-slot preconditions hold, attempt numbers cannot skip/repeat transitions, and a trajectory-wide violation remains visible after final-state repair.

The first two public development runs and their preconditions/identifier limitations are retained in the evidence archive. The third run follows a disclosed protocol amendment that removes deterministic fault/index association and uses opaque IDs. No reference policies were tuned between these runs. These results are development evidence, not untouched confirmation.

Read [research scope and next steps](DEV-049-050-RESEARCH-DIRECTION.md), [reproduction commands](DEV-049-050-REPRODUCTION.md), and [the archived raw evidence](https://github.com/FPC-effortless/TAC-OSM/tree/evidence/dev049-050-development-2026-10-17).
