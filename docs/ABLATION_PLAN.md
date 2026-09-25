# TAC-OSM Ablation Plan

**Status:** pre-registered design. Primary scientific instrument for the
integrated model.
**Dependencies:** `docs/ARCHITECTURE.md`, `provenance/COMPONENTS.md`.

---

## 1. Why ablation, not component proof

The prior methodology asked *can each primitive exist*, and answered it
through sequential gated experiments. That produced excellent controls but
left the actual question open:

> Does a mechanism contribute to useful behavior **inside the integrated
> system**?

That question is only answerable against the integrated model. Every
ablation below is therefore one architecture with controlled switches —
never a separate experiment codebase.

The distinction that matters:

> Component *existence* asks whether a mechanism can work.
> Component *responsibility* asks whether it is doing the work.

The first is necessary but cheap. The second is the whole point, and it
requires an integrated model and intervention.

---

## 2. Pre-model gates (cheap, analytical, run before training)

Only three things must be established before building. None is a
multi-stage research program.

### G1 Interface validity

The information boundary is correct. Unit-tested, not empirical.

```
router CANNOT see:
    target, answer, true edge, gold structure id,
    future outcome, oracle action, oracle mask,
    metadata uniquely encoding the correct action
```

Required: a property test asserting that the router's observable input set
is disjoint from the forbidden set. Mirrors the anti-leakage boundary in
`PNDS_MASTER_BENCHMARK.md` §7.

### G2 Representability

The intended relation is expressible *through the actual learned feature
map* — not through an oracle arm. Directly from PNDS-URP v0.4 §34, added
because `dd8f63c`'s key-blind feature map scored ideal weights at exactly 0
for every candidate while the `true_key` arm reported 1.0000.

The probe: pass analytically ideal weights through the **actual** feature
map and require a positive margin separating the gold candidate from the
best distractor, over many episodes. Must **fail** on the `dd8f63c` map and
**pass** on the `f989430` corrected map. See `test_representability.py`.

### G3 Benchmark validity

The environment requires the claimed capability. Cheap structural tests:

- gold is the **unique** satisfier (else the task is ambiguous)
- the relation is decidable from **router-visible** inputs alone
- the fixed baseline rule provably **cannot** express the relation
- exact-generated solutions are the primary metric, not teacher-forced loss

---

## 3. Primary integrated metric

The unified thesis gives one natural evaluation. Router accuracy, LM loss,
and gate accuracy are **diagnostics** — never primary endpoints, per the
two loss/competence divergences recorded in §6 of `ARCHITECTURE.md`.

| Metric | What it tests |
|---|---|
| **Capability** | exact task success, class-balanced |
| **Computation** | how much computation was actually executed |
| **History scaling** | behavior as irrelevant history grows |
| **Transfer** | does it hold when relevant structure changes |
| **Intervention** | does changing selected state/structure change the outcome |
| **Verification** | does it detect and repair incorrect computation |

The PNDS claim, tested directly:

```
C_executed  ≈  f(|R|)          intended
C_executed  ≈  f(|H|)          conventional baseline
```

where `R` is relevant structure and `H` is accumulated history. Measured as
the joint capability/cost curve, never as a single speedup number. History
levels frozen before the confirmatory run, capability parity margin
declared in advance.

---

## 4. Component ablation matrix

One architecture, switches flipped.

| Model | State | Routing | Structure | Verifier | Repair |
|---|---|---|---|---|---|
| Baseline | — | — | — | — | — |
| +State | ✓ | — | — | — | — |
| +Routing | ✓ | ✓ | — | — | — |
| +Structure | ✓ | ✓ | ✓ | — | — |
| +Verifier | ✓ | ✓ | ✓ | ✓ | — |
| Full TAC-OSM | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## 5. Mechanism-specific interventions

Simple removal distinguishes *the component exists* from *the component is
responsible*. These go further.

### 5.1 Persistence

| Intervention | Expected if persistence is load-bearing |
|---|---|
| persistent state | baseline |
| shuffled state | degradation |
| reset state | degradation |
| corrupted state | degradation, dose-response |
| random state | collapse to baseline |
| wrong-key state | collapse to chance |

Stage 3b-r2 already measured the dose-response:
0.8667 → 0.3500 → 0.1000 → 0.0000 at target corruption 0 / 0.25 / 0.5 / 1,
with key-reset collapsing accuracy to exactly chance (0.1250). That is the
reference standard for what a positive result looks like.

### 5.2 Routing

| Router | Purpose |
|---|---|
| learned | the actual mechanism |
| static similarity | cheap shortcut control — Stage 2b showed it can capture most headroom for free |
| random | floor |
| oracle | ceiling reference, not a competitor |
| full-context | conventional baseline; the real comparison for the thesis |

The static arm is the one that matters. Stage 2b learned this the hard way:
at 8 candidates the static scorer alone reached 0.7133 against an oracle
ceiling of 1.0, capturing ~71% of headroom without any learning. A routing
result that doesn't beat static is not a routing result.

### 5.3 Structural execution

| Variant | Purpose |
|---|---|
| learned structure | the actual mechanism |
| random structure | floor |
| oracle structure | ceiling |
| flattened computation | does structural routing beat dense compute |

### 5.4 Verification

| Variant | Purpose |
|---|---| 
| no verifier | floor |
| final-only verifier | current TAC lineage |
| path verifier | `V_path(z_1..z_n, y)` vs `V_final(y)` — currently NOT RUN anywhere |
| verifier + repair | the full loop |

The path-vs-final comparison is genuinely open: the master benchmark §19
requires it and no repository has run it.

---

## 6. Failure attribution

The three-model design localizes where a failure comes from.

```
Baseline            → capability ceiling
   │
TAC-OSM             → actual system
   │
Oracle TAC-OSM      → mechanism diagnostic
```

An oracle-assisted variant, where certain mechanisms can consult oracle
information as a diagnostic control only, separates:

> architecture vs representation vs routing vs execution vs optimization
> vs environment

If TAC-OSM fails and Oracle TAC-OSM also fails, the bottleneck is the
environment or the architecture. If Oracle succeeds where TAC-OSM fails,
the gap is in a specific mechanism — and that is where EXP009C, REAL012-A,
Stage 4W, and CDL distillation become targeted investigations into the
revealed failure rather than sequential prerequisites.

---

## 7. Config schema

One architecture, controlled switches. No per-experiment code paths.

```yaml
model: tac_osm_v0

state:
  enabled: true
  write: true
  intervention: persistent      # persistent|shuffled|reset|corrupted|random|wrong_key

router:
  type: learned                 # learned|static|random|oracle|full_context

structure:
  type: learned                 # learned|random|oracle|flattened

verifier:
  type: path                    # none|final|path
  repair: true

history:
  levels: [1000, 4000, 16000, 32000, 64000, 100000]
  parity_margin: pre_declared
```

`state.enabled: false` produces `TAC-OSM − persistence`.
`router.type: random` produces `TAC-OSM − learned routing`. Every row of
the matrix in §4 is this config with fields flipped.

---

## 8. Interpretation discipline

An ablation result establishes mechanism *responsibility within this
system*. It does not establish:

- that the mechanism generalizes beyond the tested environment
- that the architecture is the right one
- that the effect survives larger scale or natural data
- that any positive system-level result has been obtained

And specifically, per the master benchmark: **routing accuracy is not
sufficient evidence of execution.** A positive routing ablation with no
intervention effect on outcomes establishes nothing about the executed
computation.
