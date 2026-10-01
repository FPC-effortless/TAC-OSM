# TACOSM-PERSISTENT-SEMANTIC-REP-005 RESULT

## Run provenance

- workflow run: `36654078573`
- job: `109694456600`
- code head: `0c4c51e8b20c99210f3a2c01e9cb9362469dde24`
- branch: `research/rep005-persistent-semantic-routing`
- full test suite: **834 passed**
- baseline current test count: **834**
- artifact: `TACOSM-PERSISTENT-SEMANTIC-REP-005`
- artifact id: `11071707586`
- artifact zip SHA-256: `98514aaf064f84f2fda2651047da4855c97056c56ee41cf6d88452985a52a65e`

## State-boundary verification

The experiment wrote the semantic requirement to `goal:semantic` at step t,
then exposed it at t+1 through an address-only query.

Across the five learned runs:

- mean inspected state slots: **1.0**;
- mean state pool size: **1.0**;
- exact graph execution: **1.000**;
- reset intervention: **160/160 fail-closed**.

The query payload did not contain the five-bit semantic requirement; it contained
only the state address.

This establishes that the tested router can consume a semantic requirement from
the temporal persistent-state interface rather than from the public query text.

## Routing result

| Seed | Learned B Top-1 | No-learning C Top-1 | B − C | B training successes | B updates |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.1133 | 0.0508 | +0.0625 | 45 | 45 |
| 1 | 0.1250 | 0.1250 | +0.0000 | 52 | 52 |
| 2 | 0.0000 | 0.0000 | +0.0000 | 0 | 0 |
| 3 | 0.5156 | 0.2773 | +0.2383 | 212 | 212 |
| 4 | 0.0625 | 0.0586 | +0.0039 | 23 | 23 |
| **Pooled** | **0.1633 (209/1280)** | **0.1023 (131/1280)** | **+0.0609** | **332/2560** | **332** |

Chance with eight candidates is 0.1250.

The learned arm is above chance when pooled, while the frozen no-learning arm
is below chance. The learned-minus-control improvement is 6.09 percentage
points pooled.

Mean target rank:

- learned B: **4.5797**;
- no-learning C: **4.8477**.

The seed spread is substantial. Seed 3 supplies most of the aggregate gain;
seed 2 produces no successful training update and no routing successes.

## Analytic representability

The exact state-conditioned semantic matcher achieved:

- Top-1 recall: **1.000** for all five seeds;
- mean target rank: **1.000**;
- hard-negative margin: **4.000**.

This separates representability from learning. The state value plus explicit
candidate topology contains enough information for exact selection; the
registered learned router only partially recovers that relation.

## Causal reset

After the semantic state item was written and advanced to its declared read
boundary, the reset arm cleared state before routing.

All 160/160 reset trials failed closed. This is an interface-integrity result,
not a performance score. It shows the router did not silently recover the
semantic requirement from the query or another hidden field.

## Research conclusion

REP-005 establishes the first bounded positive result in this ladder for the
combined path:

`persistent write -> temporal boundary -> addressed read -> learned program routing`

The result supports the narrower architectural claim that the current system
can transport task-relevant semantic state across a temporal boundary and make
that state available to a program-selection mechanism.

It does **not** establish long-horizon memory quality. The delay is only one
boundary, the state pool contains one item, and the address is explicitly
provided by the query.

The learned routing remains unstable across seeds. Therefore the experiment
should be treated as an architectural integration result, not as evidence that
the learning problem is solved.

## Cost

The routing calculation is unchanged from REP-003:

`C_route = 8*5 + H*(16*7 + 8*16 + 8)`

At H=8, this is **2,024 MACs** before nonlinear functions and software overhead.

State addressing inspected one state item in the registered setup. This is
reported separately from routing MACs.

Candidate routing remains O(H). No sublinear retrieval result is present here.

## Next experiment

The next missing PNDS boundary is not another router optimizer variant. It is
semantic state addressing.

REP-006 should use multiple persistent state items with randomized opaque
addresses. The public query should describe the required state semantically
without naming its address. The system must first identify the relevant state
item, then route over explicit candidate programs.

This would turn the current:

`known address -> state read -> program route`

into:

`query -> relevant-state address -> state read -> program route`

and would test the full relevance-routing structure more directly.

## Frozen evidence

REP-004, REP-003, REP-002, REP-001, and all prior C5 evidence remain separate
historical measurements and were not overwritten.