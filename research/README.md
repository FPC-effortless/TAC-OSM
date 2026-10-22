# TAC-OSM active research index

Updated 2026-10-21 on the current **draft research branch**. Source, results and
model artifacts are pushed for review; nothing here implies a merge, default
runtime change, commercial release or complete predictive-world-model qualification.

| Study | Implemented result | Qualification boundary |
|---|---|---|
| [ME035](mathematical-engineering/me035/RESULTS.md) | Improved native normalization/slow-state/initialization; AR/native composite | Development only; original precision exception preserved |
| [ME036](mathematical-engineering/me036/RESULTS.md) | Patched native full 96/192/336/720 forecasts | 0.57% AR gain missed 2% gate |
| [ME037](mathematical-engineering/me037/RESULTS.md) | Context-integrated native model with bounded horizon correction | Development gate passed; household gate failed |
| [ME038](mathematical-engineering/me038/RESULTS.md) | New native state transition with error-corrected associative memory | Tetouan power target passed; CPU/legacy targets failed |
| [ME039](mathematical-engineering/me039/RESULTS.md) | Explicit compiled native training path and original repository-owned model delivery | Small training gain missed target; retain original serialized serving |
| [ME040](mathematical-engineering/me040/PROTOCOL.md) | Six paired native updates with unchanged coefficient capacity | Changed predictions; new trained variant, not equivalent execution |
| [ME041](https://github.com/FPC-effortless/TAC-OSM/pull/306) | 88 complete portfolio fits on Tetouan; CDL best at equal updates, context best at equal CPU | No new candidate cleared both gates |
| [ME042](https://github.com/FPC-effortless/TAC-OSM/pull/307) | 88 cross-domain ETTh1 fits; context/native win different budgets | Historical DEV only; eight treatments failed combined gates |
| [ME043 / B](mathematical-engineering/me043/RESULTS.md) | 160 authentic complete-model fits; context-seeded conditional native expert and scheduled expert learning | ~2.9x faster training than first adaptive design; cross-domain accuracy gates failed |
| [ME044](mathematical-engineering/me044/RESULTS.md) | 80 complete fits; separate multiscale native decoder; optional1.414x exact compiled inference | Oil-temp improvement over practical references; combined gates failed |

Historical runtime choices: ME037 remains the preferred recorded candidate for the legacy
benchmark matrix. ME038 is the newer task-specific native architecture, with a
passed one-city temporal power target and unresolved compute cost. Use original
ME038 full serialized models for serving; ME039's alternate export was slower,
and its separate freeze construction failed exact replay and was rejected.

[Architecture-first master plan](ARCHITECTURE-RESEARCH-PLAN.md) ·
[Claims](../docs/CLAIMS.md) · [Evidence ledger](../docs/EVIDENCE_REGISTER.md) ·
[Real-world/parity rules](REAL-WORLD-EVIDENCE-RULES.md) ·
[Original model and numerical-record downloads](model-artifacts/README.md)

The download directory now includes actual original trained model bytes and all
new ME039 optimizer/gradient records with SHA-256 verification. Git retains the
archives, and CI provides an additional 30-day Actions artifact. See actual
repository/download receipts in ME039. The older complete 1.05 GB ME038 evidence
archive remains a verified local delivery because release-asset upload credentials
were rejected; its exact availability is documented rather than implying hosting.

These forecasts remain components of the broader TAC/CDL/CASM research ambition.
They do not prove action utility, causal interventions, value of inspection,
arbitrary-lifetime streaming or native-memory architectural superiority. Detailed
mechanistic attribution stays deferred while complete models are engineered.


## Mandatory scientific governance reset (2026-10-22)

[Read the capability-first research reset](CAPABILITY-FIRST-RESEARCH-RESET-2026-10-22.md)
and the established [equal-resource rules](REAL-WORLD-EVIDENCE-RULES.md)
before launching another architecture comparison. Automated
`scripts/validate_research_parity.py` **must pass before and after training**;
missing/mismatched receipts block further experiments and claims. Most legacy
ME comparisons are unmatched practical references, not scientific proof of
architectural superiority. New work must reuse known research and pursue a
single integrated, real-data capability question rather than micro-ablations.
