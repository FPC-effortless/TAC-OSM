# TACOSM-PERSISTENT-SEMANTIC-REP-005

Status: PREREGISTERED — result pending.

## Purpose

REP-005 moves the semantic requirement from the public query into temporal persistent state. The candidate executable topology remains explicit.

The tested loop is:

world write -> causal delay -> addressed state read -> semantic router -> exact execution

This is the first experiment in this ladder that exercises the repository's temporal persistent-state boundary together with the explicit-program routing boundary.

## Task

Each episode samples one target executable topology from the fixed seven-edge substrate and derives its five-bit semantic dependency signature.

The signature is written to public address `goal:semantic` at step `t`.
The state becomes readable only at `t+1`.
The decision query at `t+1` contains only `\tgoal:semantic`.

Thus the semantic target is not present in the query payload. The router must obtain it through the persistent state interface.

Each candidate exposes only its own executable topology; candidate action indices contain no target information.

Exactly one candidate in each pool has the queried semantic signature.

## Protocol

- seeds: 0, 1, 2, 3, 4;
- training episodes: 512 per seed;
- held-out episodes: 256 per seed;
- candidates: 8;
- state delay: 1 decision boundary;
- state address: `goal:semantic`;
- query payload: address only;
- candidate topology width: 7;
- query semantic width: 5;
- candidate encoder: 7 -> 16 tanh -> 8;
- query encoder: 5 -> 8;
- learning rate: 0.01;
- margin: 0.1;
- executor: exact explicit graph executor.

## Conditions

A. Analytic semantic matcher using the state-read value — representability witness.
B. Learned persistent-state graph router.
C. Identical router with no parameter updates.
D. State-reset intervention: write occurs, then state is cleared before read; router must fail closed.
E. Oracle.

## Primary diagnostics

1. Held-out Top-1 recall of B.
2. Comparison of B versus C.
3. State addressing inspection count and state pool size.
4. Exact edge execution.
5. Reset fail-closed rate.

## Causal boundary

The reset intervention is intentionally different from a no-state training control. It keeps the task, candidate pool, query address, and router fixed, then removes the state value after the world write but before the decision boundary.

A 100 percent fail-closed rate is expected by interface contract. It is not a capability metric.

## Cost accounting

The learned routing calculation remains:

`C_route = 8*5 + H*(16*7 + 8*16 + 8)`

At H=8, this is 2,024 MACs before nonlinear operations and software overhead.

State addressing uses `TemporalPersistentState` plus `StateAddressor`. The current temporal store resolves the named address directly; the experiment reports addressing work separately from router MACs.

REP-005 does not establish sublinear candidate retrieval. Candidate routing is still O(H).

## Falsifiers

- state value is unavailable at the declared read step despite the staged write;
- analytic state-conditioned matching fails;
- B cannot exploit the state value beyond the no-learning control;
- state reset does not cause fail-closed routing;
- candidate topology and exact execution disagree;
- prior C5/REP-001/REP-002/REP-003/REP-004 evidence is modified.

## Interpretation boundary

Even a positive result would establish only that the current synthetic graph router can consume a temporally persisted semantic requirement through an explicit address boundary.

It would not establish general persistent intelligence, long-horizon memory, semantic retrieval at scale, selective hardware execution, PLM/PNDS validity, or general intelligence.

The experiment is designed to test one architectural claim at a time and preserve the earlier separation between representation, state, selection, execution, and learning.