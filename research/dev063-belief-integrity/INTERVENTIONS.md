# Prospective intervention implementation details

Frozen before opening the new test corpus. Main fitting/protocol unchanged.
Evaluate all140 fitted checkpoints on their four primary recorded cells; smoke
uses only its disjoint validation cells. Force each model's own recorded actions
and receipts, checking baseline posterior/choice parity before interventions.
No alternative action is executed and no outcome-improvement claim follows.

Compare identity-reinserted, exact public, cyclic-shuffled learned and50%-uniform
blurred joint posteriors at the semantic head, plus clock origin+10000 and clock
scale*1000. Identity reinsertion must preserve the baseline. Relative event arms
must preserve logits and choices on clock-only changes within1e-8; absolute arms
are controls, not required to pass this symmetry. No weights or decoder change.

Shuffle donors cyclically among active worlds with the same PUBLIC dependency
graph, preserving task context while changing health belief. Decision-relevant
eligibility means exact public teacher actions differ between recipient/donor
health posteriors under that same public graph and prices. Cache exact teacher
policy calls without rounding probabilities. Report all flips and eligible flips;
blur is an off-training-distribution diagnostic, not proof of realistic utility.
Near-zero exact flips can mean fidelity. No posthoc flip-rate promotion threshold
is introduced; sensitivity remains descriptive alongside structural/operational
gates. Scalar mean/p95/p99 KL, not action flips alone, assesses representation.

Cap1200CPU seconds including reconstruction/teacher eligibility/inference/logging.
Stream every query's donor/index/choices/deltas to compressed JSONL and preserve
partial results on failure/cap. Incomplete checkpoint/cell coverage blocks integrity
qualification. All20seeds share the same64physical starts across histories; query
counts do not become independent physical or calibration trials.
