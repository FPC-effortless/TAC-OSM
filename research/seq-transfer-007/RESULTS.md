# SEQ-TRANSFER-007 — teacher-supervision ablation results

**Status: COMPLETE EXPLORATORY VALIDATION ONLY; no model promotion.**
Training took place 2026-10-19 on GitHub Actions CPU; all 12 fixed fits
finished with 400 updates each. Both workflow jobs passed:
[run 39302634102](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39302634102).
Retained artifact: `seq007-original-teacher-ablation-pilots` (12 original
weights/checkpoints and `pilot.json`). No held-out TEST data were opened.
Source train/validation hashes are checked against the original SEQ006 pilots.

## Primary measured result

At **both L64/E4 and L128/E4**, for both fresh optimization seeds
(690810 and 690811), the TAC-inspired entity-local gated recurrence and
simpler independently trained routed GRU obtained exactly **100% correct
final predictions on each** of the late-overwrite, repeated-toggle and
pair-XOR tasks, **with and without intermediate exact-state teacher labels**.

| Architecture | L128 overwrite without/with teacher | L128 toggle without/with | L128 pair XOR without/with |
|---|---:|---:|---:|
| TAC-inspired gated | 100/100% | 100/100% | 100/100% |
| Entity-routed GRU | 100/100% | 100/100% | 100/100% |
| Global GRU | 52.25/50.25% | 48.75/46.75% | 48.00/52.25% |

Global-GRU values are arithmetic means across two seed runs, not
confidence intervals. The same 600 validation examples per horizon
are shared among every arm; they are **not** 1,200 statistically
independent physical systems. Each task has 200 examples/length.

## Interpretation

1. **Exact intermediate-state training labels were NOT necessary for
   these bounded tasks** under the fixed final-output classification
   supervision, 400 updates and public-ID routing prior. The stronger
   claim of autonomous posterior-free world discovery is not supported:
   final state labels come from a deterministic symbolic simulator.
2. Shared entity-local addressing appears more important than
   specialized TAC operator complexity: a smaller conceptual
   entity-routed GRU exactly matched TAC on every validation task.
3. The global-GRU control stayed near chance despite separate
   dense-teacher supervision, but this experiment cannot establish
   universal architectural superiority or optimal hyperparameters.
4. Training-length-to-L128 generalization is still only a validation
   result with two optimizer seeds. The L256/L512/E6 TEST belongs to
   separate preregistered SEQ006C; no outcome here may retune that
   running confirmatory study.

## Limits and generator caveats

- Query labels are balanced, and event identity is publicly supplied.
- Operations are fixed SET0/SET1/TOGGLE/NOOP symbols with a known
  positional source generator, not learned action discovery.
- The XOR task applies **two toggles to each queried entity**, which
  cancel; the task tests role-binding and final XOR composition but
  does not force a nontrivial variable toggle parity on both entities.
- Task-specific informative event positions are constrained by the
  generator; broader adversarial timing, aliases, and noisy evidence
  have not been evaluated.
- Models do not receive hidden state at inference, but final state
  labels remain supervised at training; no economic decision/repair
  tests have been conducted.
- Model parameter counts differ between architectural families, but
  match exactly within each same-architecture loss ablation.
- Initial pilot seeds are insufficient for confirmatory attribution.

## Next research decision

Shift the main mechanism hypothesis toward **learned update and
role-binding operators without exact state supervision**, tested against
entity-routed GRU under deliberately less scripted and less
predictable event timing. The next independent registration should
include noncancelling pairwise updates, identity aliases/rebindings,
unseen entities, irrelevant noisy receipts and optional observations,
with a fresh unexposed benchmark. Do not use this report to claim
CASM-specific superiority or replace previously accepted scientific gates.

**Data provenance:** All original raw outputs are preserved in
[CI artifact](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39302634102).
The branch remains research-only and unmerged.
