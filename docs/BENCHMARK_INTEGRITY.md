# Benchmark Integrity Checklist

The permanent record of the defects that could have made a negative result
uninterpretable. Every entry is a failure mode that was *found by audit*, not
by a failing test — in most cases no test could have caught it, because the
benchmark reported a plausible-looking number while being wrong.

**These are safeguards for every subsequent TAC-OSM environment, not historical
debugging notes.** When a new environment is written, run this list against it.

---

## The six defects

### 1. Wrong operands in the relevance circuit

**Symptom.** The circuit computed agreement of the reference against itself
and the descriptor against itself, rather than reference against descriptor.

**Consequence.** Output was identical for a satisfying and a violating
candidate. Execution was a **tautology**, not a cross-check, so `V_t` could
never reject anything and the loop's verification stage measured nothing.

**Found by.** Reading the circuit constructor against the interface layout.

**Safeguard.** The circuit is now unit-tested against the environment's own
`satisfies_relation` over 50 seeds, for both directions
(`test_relevance_circuit_is_one_for_the_relation_satisfier`). Verification is
also asserted to *reject* a correct execution when its reference is wrong
(`test_a_correct_execution_verifies_on_every_family`).

---

### 2. Exactness unreachable through the learned parameter

**Symptom.** `_exact_alpha_for` returned `[1.0, 1.0]`, which passed through
`softplus` and became 1.313 — never 1.0.

**Consequence.** Disagreement scored −0.313 and agreement chains accumulated
1.313², so the circuit was never in the intended exact Boolean regime. The
scoring semantics were mathematically inconsistent with the relation.

**Found by.** Analytic inspection of the parameter path.

**Safeguard.** Exactness is now a distinct **mode** (`alpha is None`) with its
own `_apply_exact`, not a value the learned parameter can approach. Every
executed output is asserted to be exactly 0.0 or 1.0
(`test_the_executed_output_is_boolean_for_the_relevance_circuit`).

---

### 3. Stale gold bookkeeping after shuffle

**Symptom.** `_finalise` shuffles and *rebuilds* every candidate, but
`RelationalTaskSpec.gold_index` and `gold_descriptor` were computed by the
caller before the shuffle.

**Consequence.** `target_action` pointed at a candidate that did not satisfy
the relation. **Every oracle arm scored ~0 with the circuit correct** — the
most dangerous failure in the set, because it looks like a broken mechanism
when the mechanism is fine.

**Found by.** The oracle preflight, once defect 1 was fixed.

**Safeguard.** Both fields are re-derived from the placed candidate inside
`_finalise`. `test_gold_index_and_descriptor_agree_after_shuffle` asserts that
`target_action` is the relation's *unique* satisfier, over 50 seeds.

> **Rule.** Gold labels must be derived **after** candidate placement and
> randomisation, never before.

---

### 4. Verification trace carried padding

**Symptom.** The trace was padded to `max_nodes` with zeros.

**Consequence.** Path verification read 25 phantom nodes at 0.0 and failed
every step, so `writes = 0` while the computation was sound.

**Found by.** Comparing `writes` against the oracle's correctness.

**Safeguard.** The trace is `values[:active_count]`, and
`test_the_executed_trace_carries_active_nodes_only` pins
`len(trace) == active_count + 1`.

> **Rule.** Verifier traces must contain only executed nodes, never padded
> sentinel values.

---

### 5. Fixed tolerance unrelated to the output range

**Symptom.** `path_tolerance = 0.25`, applied to a Boolean circuit whose EQ
nodes are legitimately 0 or 1 while the output is 0.

**Consequence.** Any legitimate non-satisfying candidate failed path
verification, so the verifier conflated "wrong answer" with "inconsistent
computation".

**Found by.** Reading the tolerance against the executed output distribution.

**Safeguard.** Tolerance is derived from the computation's own range
(`_path_tolerance_for`).

---

### 6. The executor's reference preferred the query bits over the state address

**Symptom.** `_reference_for` returned the query's public bits whenever they
existed. For `replay`, which carries **both** public bits and a state address,
that is the wrong vector.

**Consequence.** The executor computed a relation the environment never
defined. Gold still satisfied the *true* relation, but the circuit said it did
not, so `V_t` rejected a correct execution ~75% of the time on `replay` and
suppressed the legitimate write. **Accuracy was unaffected** — the oracle
still scored 1.0 — so the defect was invisible in the headline number and
appeared only as a low write count.

**Found by.** The §34 representability audit, while designing the extended
feature basis. No test caught it; the audit did.

**Safeguard.** The reference is now address-first. Four tests pin the
per-family reference (`test_replay_circuit_uses_the_written_vector_not_the_
query_bits`, `test_lookup_circuit_uses_the_written_vector`,
`test_relational_circuit_uses_the_query_bits`) and one pins the end-to-end
consequence (`test_a_correct_execution_verifies_on_every_family`).

> **Rule.** When a task carries both a public observation and a state address,
> the address names where the target lives. Prefer the address.

---

## The seventh finding: the hypothesis-class gap

Not a defect in the code — a defect in the *evidence* the gate produced.

**Symptom.** `check_representability` passed, and the learned router scored
0.0833.

**Cause.** The gate ran a **per-episode** ideal weight vector on
`relational` only. A per-episode ideal is necessary but not sufficient: the
ideal vector can rotate with the marked positions, which change every episode,
while a learned linear router has one fixed vector. The persistence families
were never audited at all.

**Measurement.** Under a shared weight vector, `relational` separated at
+1.4754 while `state_lookup` reached **−0.2353** and `replay` **−2.3210** — a
negative margin, meaning the ideal vector actively prefers a distractor. The
unmarked agree-rate for gold was 0.04–0.06, confirming the anti-signal is
nearly deterministic. The gap was in the hypothesis class, not the optimiser.

**Fix.** The basis gained a context-gated slot block
(`slot_gated_s_j = present(s) · context_j · [value(s)_j == descriptor_j]`),
gated to the addressed slot only, and the gate now runs the **shared-weight**
form across all three families. With `gated_agreement = 1.0` and
`slot_gated = 2.0` the margin is +2.0 / +4.0 / +2.0 with zero ties. The
learned arm moved from 0.0833 to 0.452 over 500 steps, monotonically and
stably across seeds.

**Safeguard.** `test_shared_weights_detect_a_hypothesis_class_gap`,
`test_analytic_weights_separate_gold_on_every_family`, and
`test_the_gate_fails_without_the_gated_slot_block`.

> **Rule.** A representability gate must use a weight vector **shared across
> every episode**, and must cover **every task family**. A per-episode ideal
> proves nothing about what one learned vector can express.

---

## The standing rules

1. **Oracle accuracy must be established before interpreting any learned
   result.** If the oracle does not reach 1.0, the environment is ambiguous
   and no learned-arm number means anything.
2. **Gold labels are derived after placement, not before.**
3. **Verifier traces contain only executed nodes, never padded sentinels.**
4. **Verification must agree with success.** `success = True` with
   `verification.passed = False` means the verifier is computing a different
   relation than the environment scores against.
5. **A representability gate uses shared weights, across all families.**
6. **A negative learned result is attributed only after the gate passes.**
   Until then, it is uninterpretable, not evidence about learning.

---

## How to run the list against a new environment

1. Run the oracle arm. Require 1.0.
2. Run the audit on the *actual* basis, with a *shared* vector, on the
   *actual* episodes, for *every* family.
3. Assert `target_action` is the unique satisfier, post-shuffle.
4. Assert the trace length equals the active node count.
5. Assert `success` and `verification.passed` agree on the oracle arm.
6. Only then read a learned-arm number.


---

## The eighth finding: modality payload / metadata collision

**Symptom.** An image modality encoded both payload bits and an entity-ID stripe in the same spatial rows. Image bits 4–5 were placed in rows 1–3 while the entity-ID stripe occupied rows 0–3; image bits 6–7 were placed elsewhere and remained clean.

**Consequence.** The benchmark created an unintended shortcut/collision in which some queried payload bits were entangled with metadata. The resulting operator-specific error pattern was initially liable to be misread as a state-content or CASM weakness.

**Found by.** The AND state-versus-CASM diagnostic and inspection of the actual generator geometry. Image bits 4–5 had materially degraded recovery while bits 6–7 were recovered perfectly; conditional AND execution error after correct bit recovery was only ~2.3%.

**Safeguard.** Every multimodal benchmark must explicitly declare which fields are encoded in each modality and must test that payload support does not overlap metadata support. For spatial modalities, the audit must inspect the actual masks/regions rather than relying on comments or intended layout.

> **Rule.** A modality may not silently carry task metadata in the same support as the payload. Metadata channels must be explicit scaffolds or absent; unintended payload/metadata overlap invalidates mechanistic interpretation of affected results.

The clean E2E-008 successor removes the image entity stripe, removes entity dependence from text/audio encodings, and separates image payload regions from the former metadata support.