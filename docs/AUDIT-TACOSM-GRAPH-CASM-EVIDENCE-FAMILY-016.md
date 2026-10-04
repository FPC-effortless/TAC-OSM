# G-CASM-016 — Evidence Family Comparison

**Status: PREREGISTERED / DESIGN ONLY**

G-CASM-014 showed that adaptive row placement can approach the finite-domain
collision floor without materially restoring large-M capability at B=8.
Therefore the next variable is the observation channel itself.

## Scientific separation

This experiment distinguishes three quantities:

1. **Information content** — how much uncertainty about the hypothesis is removed.
2. **Acquisition efficiency** — information removed per environment execution/read cost.
3. **Verified utility** — whether the acquired evidence improves downstream exact
   computation under a fixed candidate-execution budget.

A channel that wins only (1) is not an architectural winner.

## Registered evidence families

### scalar_output

A single ordinary environment response:

    x -> f(x)

Cost: one exact execution.

### paired_output

One base input and one fixed counterfactual perturbation:

    (x, x xor delta) -> (f(x), f(x xor delta))

with delta fixed before measurement.

Cost: two exact executions.

The perturbation is not chosen adaptively. This prevents an additional action
selection degree of freedom from becoming an unregistered source of advantage.

### quad_output

A fixed four-row local neighborhood produces four external binary responses.

Cost: four exact executions.

This is intentionally expensive. It provides a clean reference for whether
simply concatenating scalar evidence is preferable to richer structured
semantics.

### activation_trace_ceiling

The G-CASM-015 instrumented six-bit internal activation trace.

This is an explicit upper-bound channel rather than a deployable assumption.
It answers whether internal computational evidence exists that would be useful
if an environment exposed it.

## Expected scientific outcomes

There are four interesting cases:

### A. External structured evidence beats scalar evidence

This is the strongest outcome. It would establish that the architecture can gain
information through action choice without relying on privileged internal state.

### B. Internal trace wins but external structure does not

This indicates that the missing resource is not merely richer information but a
deployable interface to computational structure.

### C. More bits do not improve verified utility

This would show that entropy alone is insufficient. The evidence needs to be
aligned with downstream hypothesis discrimination or executable computation.

### D. Structured evidence loses after cost normalization

Then the problem is an information-density problem: the channel contains useful
information but acquires it too expensively.

## Consequence for PLM

The preferred next abstraction is therefore not "more features."

It is:

    Probe -> EvidenceCompiler -> BeliefUpdate -> Verification

The EvidenceCompiler should transform raw observations into the smallest
validated representation that preserves distinctions useful for future actions.

This becomes an architectural mechanism in its own right:

    raw evidence
       ↓
    structural compression
       ↓
    sufficient evidence state
       ↓
    reusable persistent experience

The compressor must be evaluated by downstream verified utility, not by
reconstruction error alone.

## Non-claims

016 does not establish learned probe selection, asymptotic scaling, general
active learning superiority, multimodal competence, or hardware speedup.
