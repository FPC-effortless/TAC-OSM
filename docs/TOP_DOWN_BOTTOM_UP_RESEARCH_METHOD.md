# Top-Down Assembly with Bottom-Up Ablation

## Purpose

This is the default method for introducing a new architectural mechanism into
TAC-OSM / PLM.

It prevents two opposite failure modes:

- building isolated modules that never prove they contribute to useful
  end-to-end behavior;
- changing several mechanisms at once and then attributing the result to the
  newest idea.

The method is therefore:

**assemble top-down, isolate bottom-up.**

## 1. Top-down assembly

Start from the complete competitive unit:

```
representation
    -> selective computation
    -> action
    -> outcome
    -> evaluation
    -> improvement
```

Instantiate the surrounding system with standard, already-understood
primitives wherever possible:

- standard observation/representation interface;
- standard persistent-state interface;
- standard router or policy;
- standard execution primitive;
- standard environment outcome;
- independently checked verifier;
- standard optimizer/training procedure.

Introduce exactly **one novel mechanism** at the architectural boundary.

The novel mechanism must have a named interface, a declared state/input/output
surface, explicit computational cost, and an explicit failure mode.

### Top-down acceptance test

The assembled model must be executable end to end before any component claim
is promoted.

A passing integration test demonstrates only that the complete causal path is
operational. It does not prove that the new mechanism helps.

## 2. Novel-block boundary

The novel block is the only intended change between the top-down reference and
the nearest standard control.

Freeze:

- benchmark;
- train/validation/test identities;
- environment;
- evaluation generator;
- optimizer family;
- training schedule;
- policy/executor architecture;
- parameter budget;
- observation interface;
- verification semantics.

Measure work explicitly rather than asserting equality.

The novel block may have additional internal state only when that state is the
mechanism under test. Its state footprint and update/read work must be counted.

## 3. Bottom-up ablation ladder

Every novel block gets a decomposition ladder appropriate to its structure.

The minimum ladder is:

1. **absence:** remove the block;
2. **minimal form:** replace it with the simplest standard mechanism;
3. **intermediate forms:** remove one internal capability at a time;
4. **full form:** the registered novel block;
5. **causal interventions:** reset, shuffle, corrupt, freeze, or otherwise
   replace the block's output/state without changing the trained surrounding
   model.

For a temporal state mechanism, the natural ladder is:

```
no state
-> one timescale
-> two timescales
-> full multi-timescale state
```

plus state-reset and state-swap interventions.

For a routing mechanism, use:

```
full routing
-> reduced representation
-> reduced candidate set
-> randomized routing
-> oracle/ceiling routing
```

For a verifier/repair mechanism, use:

```
no verifier
-> final verifier
-> path verifier
-> bounded repair
-> verifier + repair with independent semantics
```

The exact ladder is preregistered before confirmation.

## 4. Causal isolation requirement

A novel block is not considered load-bearing merely because:

- its internal metric improves;
- training loss falls;
- the router gets more accurate;
- information gain increases;
- a teacher agrees with it;
- a verifier reports more positives.

The block must affect the registered exact capability endpoint or a
pre-registered mechanism endpoint.

The strongest causal intervention is to hold the current input fixed and
replace only the state/output produced by the novel block.

## 5. Matched-capacity rule

A richer novel block cannot win merely because it receives:

- more parameters;
- more hidden dimensions;
- more learned state;
- more training steps;
- more candidate evaluations;
- more privileged supervision.

At least one of the following must hold:

- parameter count is exactly matched;
- compute is exactly matched;
- the difference is explicitly measured and included in the interpretation.

A positive result under unequal capacity is a capacity result until matched.

## 6. Benchmark design rule

The benchmark must make the mechanism necessary.

For memory/adaptation:

```
same current observation
+
different histories
=
different optimal actions
```

For representation discovery:

```
same superficial descriptor
+
different executable/causal structure
=
different correct computation
```

For selective computation:

```
same final capability target
+
growing irrelevant population
=
measurable pressure on selection cost
```

For evidence acquisition:

```
same candidate set
+
different available evidence
=
different information-efficient actions
```

A current-input shortcut, target-derived metadata, candidate ordering,
population-size encoding, or evaluator-private state invalidates the relevant
claim.

## 7. Statistical structure

The primary endpoint is frozen before confirmation.

The default reporting unit for small-seed architectural experiments is the
seed-level estimate with paired evaluation instances across arms. Bootstrap
over seeds is the default uncertainty procedure unless a different valid
method is preregistered.

Pooled-trial quantiles must not be described as seed-averaged quantiles.
Materiality thresholds cannot be selected after inspecting results.

## 8. Result hierarchy

Results are interpreted in this order:

1. protocol/security integrity;
2. benchmark integrity;
3. leakage/supervision boundary;
4. representability;
5. model-state integrity;
6. degeneracy/discrimination;
7. exact capability;
8. causal ablations/interventions;
9. compute/state footprint;
10. statistics;
11. scope/generalization.

A failure at an earlier gate prevents a later scientific interpretation.

## 9. Result progression

A research lane progresses through these evidence layers:

**A. Assembly:** complete top-down path works.

**B. Contribution:** novel block improves the registered endpoint over the
matched minimal control.

**C. Necessity:** removing/resetting/replacing the block removes the observed
benefit.

**D. Robustness:** the effect survives seeds, nuisance changes and registered
capacity/compute controls.

**E. Scaling:** the effect survives increased irrelevant population, temporal
depth, or task complexity.

**F. Transfer:** the same mechanism survives a new task family or modality.

A lane cannot jump from A to F because the integrated model happens to work.

## 10. Failure interpretation

If the top-down assembly fails, classify the problem at the correct layer:

- implementation failure;
- interface failure;
- benchmark failure;
- leakage;
- representability;
- optimization;
- verification;
- state formation;
- scientific null.

Do not blame the novel mechanism until the surrounding scaffold has passed its
own controls.

If the full model works but the novel block's ablation does not change the
result, the mechanism is not yet shown to be causal. It may be redundant.

If the novel block wins only with a larger compute or parameter budget, report a
capacity/compute tradeoff rather than a mechanism advantage.

## 11. Continuity rule

Every new architectural idea must name:

- parent research lane;
- novel block;
- standard surrounding scaffold;
- nearest bottom-up controls;
- predecessor/successor experiments;
- blocker it addresses.

A new idea never closes a previous lane merely by producing a more interesting
architecture.

## 12. Current instantiation

The first explicit instantiation of this method is:

**TACOSM-PLM-TDBU-MTSK-001**

Novel block:
**Multi-Timescale Adaptive State Kernel (MTSK)**.

Standard scaffold:
**observation -> persistent state -> tanh policy/router -> fixed action
semantics -> independent environment outcome -> verifier**.

Bottom-up ladder:
**no state -> one timescale -> two timescales -> MTSK**, followed by frozen-state
reset and shuffled-state interventions.

The lane is deliberately finite and synthetic. Its result, positive or
negative, is not a claim about biology, general intelligence, multimodal
capability, asymptotic scaling, or hardware speedup.
